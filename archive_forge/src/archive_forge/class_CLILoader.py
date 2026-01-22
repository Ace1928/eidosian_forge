from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import os
import re
import sys
import types
import uuid
import argcomplete
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import backend
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import pkg_resources
import six
class CLILoader(object):
    """A class to encapsulate loading the CLI and bootstrapping the REPL."""
    PATH_RE = re.compile('(?:([\\w\\.]+)\\.)?([^\\.]+)')

    def __init__(self, name, command_root_directory, allow_non_existing_modules=False, logs_dir=None, version_func=None, known_error_handler=None, yaml_command_translator=None):
        """Initialize Calliope.

    Args:
      name: str, The name of the top level command, used for nice error
        reporting.
      command_root_directory: str, The path to the directory containing the main
        CLI module.
      allow_non_existing_modules: True to allow extra module directories to not
        exist, False to raise an exception if a module does not exist.
      logs_dir: str, The path to the root directory to store logs in, or None
        for no log files.
      version_func: func, A function to call for a top-level -v and
        --version flag. If None, no flags will be available.
      known_error_handler: f(x)->None, A function to call when an known error is
        handled. It takes a single argument that is the exception.
      yaml_command_translator: YamlCommandTranslator, An instance of a
        translator that will be used to load commands written as a yaml spec.

    Raises:
      backend.LayoutException: If no command root directory is given.
    """
        self.__name = name
        self.__command_root_directory = command_root_directory
        if not self.__command_root_directory:
            raise command_loading.LayoutException('You must specify a command root directory.')
        self.__allow_non_existing_modules = allow_non_existing_modules
        self.__logs_dir = logs_dir or config.Paths().logs_dir
        self.__version_func = version_func
        self.__known_error_handler = known_error_handler
        self.__yaml_command_translator = yaml_command_translator
        self.__pre_run_hooks = []
        self.__post_run_hooks = []
        self.__modules = []
        self.__missing_components = {}
        self.__release_tracks = {}

    @property
    def yaml_command_translator(self):
        return self.__yaml_command_translator

    def AddReleaseTrack(self, release_track, path, component=None):
        """Adds a release track to this CLI tool.

    A release track (like alpha, beta...) will appear as a subgroup under the
    main entry point of the tool.  All groups and commands will be replicated
    under each registered release track.  You can implement your commands to
    behave differently based on how they are called.

    Args:
      release_track: base.ReleaseTrack, The release track you are adding.
      path: str, The full path the directory containing the root of this group.
      component: str, The name of the component this release track is in, if
        you want calliope to auto install it for users.

    Raises:
      ValueError: If an invalid track is registered.
    """
        if not release_track.prefix:
            raise ValueError('You may only register alternate release tracks that have a different prefix.')
        self.__release_tracks[release_track] = (path, component)

    def AddModule(self, name, path, component=None):
        """Adds a module to this CLI tool.

    If you are making a CLI that has subgroups, use this to add in more
    directories of commands.

    Args:
      name: str, The name of the group to create under the main CLI.  If this is
        to be placed under another group, a dotted name can be used.
      path: str, The full path the directory containing the commands for this
        group.
      component: str, The name of the component this command module is in, if
        you want calliope to auto install it for users.
    """
        self.__modules.append((name, path, component))

    def RegisterPreRunHook(self, func, include_commands=None, exclude_commands=None):
        """Register a function to be run before command execution.

    Args:
      func: function, The function to run.  See RunHook for more details.
      include_commands: str, A regex for the command paths to run.  If not
        provided, the hook will be run for all commands.
      exclude_commands: str, A regex for the command paths to exclude.  If not
        provided, nothing will be excluded.
    """
        hook = RunHook(func, include_commands, exclude_commands)
        self.__pre_run_hooks.append(hook)

    def RegisterPostRunHook(self, func, include_commands=None, exclude_commands=None):
        """Register a function to be run after command execution.

    Args:
      func: function, The function to run.  See RunHook for more details.
      include_commands: str, A regex for the command paths to run.  If not
        provided, the hook will be run for all commands.
      exclude_commands: str, A regex for the command paths to exclude.  If not
        provided, nothing will be excluded.
    """
        hook = RunHook(func, include_commands, exclude_commands)
        self.__post_run_hooks.append(hook)

    def ComponentsForMissingCommand(self, command_path):
        """Gets the components that need to be installed to run the given command.

    Args:
      command_path: [str], The path of the command being run.

    Returns:
      [str], The component names of the components that should be installed.
    """
        path_string = '.'.join(command_path)
        return [component for path, component in six.iteritems(self.__missing_components) if path_string.startswith(self.__name + '.' + path)]

    def ReplicateCommandPathForAllOtherTracks(self, command_path):
        """Finds other release tracks this command could be in.

    The returned values are not necessarily guaranteed to exist because the
    commands could be disabled for that particular release track.  It is up to
    the caller to determine if the commands actually exist before attempting
    use.

    Args:
      command_path: [str], The path of the command being run.

    Returns:
      {ReleaseTrack: [str]}, A mapping of release track to command path of other
      places this command could be found.
    """
        if len(command_path) < 2:
            return []
        track = calliope_base.ReleaseTrack.FromPrefix(command_path[1])
        if track and track not in self.__release_tracks:
            track = None
        root = command_path[0]
        sub_path = command_path[2:] if track else command_path[1:]
        if not sub_path:
            return []
        results = dict()
        for t in self.__release_tracks:
            results[t] = [root] + [t.prefix] + sub_path
        if track:
            del results[track]
            results[calliope_base.ReleaseTrack.GA] = [root] + sub_path
        return results

    def Generate(self):
        """Uses the registered information to generate the CLI tool.

    Returns:
      CLI, The generated CLI tool.
    """
        impl_path = self.__ValidateCommandOrGroupInfo(self.__command_root_directory, allow_non_existing_modules=False)
        top_group = backend.CommandGroup([impl_path], [self.__name], calliope_base.ReleaseTrack.GA, uuid.uuid4().hex, self, None)
        self.__AddBuiltinGlobalFlags(top_group)
        loaded_release_tracks = dict([(calliope_base.ReleaseTrack.GA, top_group)])
        track_names = set((track.prefix for track in self.__release_tracks.keys()))
        for track, (module_dir, component) in six.iteritems(self.__release_tracks):
            impl_path = self.__ValidateCommandOrGroupInfo(module_dir, allow_non_existing_modules=self.__allow_non_existing_modules)
            if impl_path:
                top_group._groups_to_load[track.prefix] = [impl_path]
                track_group = top_group.LoadSubElement(track.prefix, allow_empty=True, release_track_override=track)
                top_group.CopyAllSubElementsTo(track_group, ignore=track_names)
                loaded_release_tracks[track] = track_group
            elif component:
                self.__missing_components[track.prefix] = component
        for module_dot_path, module_dir_path, component in self.__modules:
            is_command = module_dir_path.endswith(_COMMAND_SUFFIX)
            if is_command:
                module_dir_path = module_dir_path[:-len(_COMMAND_SUFFIX)]
            match = CLILoader.PATH_RE.match(module_dot_path)
            root, name = match.group(1, 2)
            try:
                for track, track_root_group in six.iteritems(loaded_release_tracks):
                    parent_group = self.__FindParentGroup(track_root_group, root)
                    exception_if_present = None
                    if not parent_group:
                        if track != calliope_base.ReleaseTrack.GA:
                            continue
                        exception_if_present = command_loading.LayoutException('Root [{root}] for command group [{group}] does not exist.'.format(root=root, group=name))
                    cmd_or_grp_name = module_dot_path.split('.')[-1]
                    impl_path = self.__ValidateCommandOrGroupInfo(module_dir_path, allow_non_existing_modules=self.__allow_non_existing_modules, exception_if_present=exception_if_present)
                    if impl_path:
                        if is_command:
                            parent_group._commands_to_load[cmd_or_grp_name] = [impl_path]
                        else:
                            parent_group._groups_to_load[cmd_or_grp_name] = [impl_path]
                    elif component:
                        prefix = track.prefix + '.' if track.prefix else ''
                        self.__missing_components[prefix + module_dot_path] = component
            except command_loading.CommandLoadFailure as e:
                log.exception(e)
        cli = self.__MakeCLI(top_group)
        return cli

    def __FindParentGroup(self, top_group, root):
        """Find the group that should be the parent of this command.

    Args:
      top_group: _CommandCommon, The top group in this CLI hierarchy.
      root: str, The dotted path of where this command or group should appear
        in the command tree.

    Returns:
      _CommandCommon, The group that should be parent of this new command tree
        or None if it could not be found.
    """
        if not root:
            return top_group
        root_path = root.split('.')
        group = top_group
        for part in root_path:
            group = group.LoadSubElement(part)
            if not group:
                return None
        return group

    def __ValidateCommandOrGroupInfo(self, impl_path, allow_non_existing_modules=False, exception_if_present=None):
        """Generates the information necessary to be able to load a command group.

    The group might actually be loaded now if it is the root of the SDK, or the
    information might be saved for later if it is to be lazy loaded.

    Args:
      impl_path: str, The file path to the command implementation for this
        command or group.
      allow_non_existing_modules: True to allow this module directory to not
        exist, False to raise an exception if this module does not exist.
      exception_if_present: Exception, An exception to throw if the module
        actually exists, or None.

    Raises:
      LayoutException: If the module directory does not exist and
      allow_non_existing is False.

    Returns:
      impl_path or None if the module directory does not exist and
      allow_non_existing is True.
    """
        module_root, module = os.path.split(impl_path)
        if not pkg_resources.IsImportable(module, module_root):
            if allow_non_existing_modules:
                return None
            raise command_loading.LayoutException('The given module directory does not exist: {0}'.format(impl_path))
        elif exception_if_present:
            raise exception_if_present
        return impl_path

    def __AddBuiltinGlobalFlags(self, top_element):
        """Adds in calliope builtin global flags.

    This needs to happen immediately after the top group is loaded and before
    any other groups are loaded.  The flags must be present so when sub groups
    are loaded, the flags propagate down.

    Args:
      top_element: backend._CommandCommon, The root of the command tree.
    """
        calliope_base.FLAGS_FILE_FLAG.AddToParser(top_element.ai)
        calliope_base.FLATTEN_FLAG.AddToParser(top_element.ai)
        calliope_base.FORMAT_FLAG.AddToParser(top_element.ai)
        if self.__version_func is not None:
            top_element.ai.add_argument('-v', '--version', do_not_propagate=True, category=calliope_base.COMMONLY_USED_FLAGS, action=actions.FunctionExitAction(self.__version_func), help='Print version information and exit. This flag is only available at the global level.')
        top_element.ai.add_argument('--configuration', metavar='CONFIGURATION', category=calliope_base.COMMONLY_USED_FLAGS, help='        The configuration to use for this command invocation. For more\n        information on how to use configurations, run:\n        `gcloud topic configurations`.  You can also use the {0} environment\n        variable to set the equivalent of this flag for a terminal\n        session.'.format(config.CLOUDSDK_ACTIVE_CONFIG_NAME))
        top_element.ai.add_argument('--verbosity', choices=log.OrderedVerbosityNames(), default=log.DEFAULT_VERBOSITY_STRING, category=calliope_base.COMMONLY_USED_FLAGS, help='Override the default verbosity for this command.', action=actions.StoreProperty(properties.VALUES.core.verbosity))
        top_element.ai.add_argument('--user-output-enabled', metavar=' ', nargs='?', default=None, const='true', choices=('true', 'false'), action=actions.DeprecationAction('--user-output-enabled', warn='The `{flag_name}` flag will no longer support the explicit use of the `true/false` optional value in an upcoming release.', removed=False, show_message=lambda _: False, action=actions.StoreBooleanProperty(properties.VALUES.core.user_output_enabled)), help='Print user intended output to the console.')
        top_element.ai.add_argument('--log-http', default=None, action=actions.StoreBooleanProperty(properties.VALUES.core.log_http), help='Log all HTTP server requests and responses to stderr.')
        top_element.ai.add_argument('--authority-selector', default=None, action=actions.StoreProperty(properties.VALUES.auth.authority_selector), hidden=True, help='THIS ARGUMENT NEEDS HELP TEXT.')
        top_element.ai.add_argument('--authorization-token-file', default=None, action=actions.StoreProperty(properties.VALUES.auth.authorization_token_file), hidden=True, help='THIS ARGUMENT NEEDS HELP TEXT.')
        top_element.ai.add_argument('--credential-file-override', action=actions.StoreProperty(properties.VALUES.auth.credential_file_override), hidden=True, help='THIS ARGUMENT NEEDS HELP TEXT.')
        top_element.ai.add_argument('--http-timeout', default=None, action=actions.StoreProperty(properties.VALUES.core.http_timeout), hidden=True, help='THIS ARGUMENT NEEDS HELP TEXT.')
        FLAG_INTERNAL_FLAG_FILE_LINE.AddToParser(top_element.ai)

    def __MakeCLI(self, top_element):
        """Generate a CLI object from the given data.

    Args:
      top_element: The top element of the command tree
        (that extends backend.CommandCommon).

    Returns:
      CLI, The generated CLI tool.
    """
        if '_ARGCOMPLETE' not in os.environ or '_ARGCOMPLETE_TRACE' in os.environ:
            log.AddFileLogging(self.__logs_dir)
            verbosity_string = encoding.GetEncodedValue(os.environ, '_ARGCOMPLETE_TRACE')
            if verbosity_string:
                verbosity = log.VALID_VERBOSITY_STRINGS.get(verbosity_string)
                log.SetVerbosity(verbosity)
        if properties.VALUES.core.disable_command_lazy_loading.GetBool():
            top_element.LoadAllSubElements(recursive=True)
        cli = CLI(self.__name, top_element, self.__pre_run_hooks, self.__post_run_hooks, self.__known_error_handler)
        return cli