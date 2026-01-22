import atexit
from copy import deepcopy
import logging
import os
import shutil
import sys
from pathlib import Path
from traitlets.config.application import Application, catch_config_error
from traitlets.config.loader import ConfigFileNotFound, PyFileConfigLoader
from IPython.core import release, crashhandler
from IPython.core.profiledir import ProfileDir, ProfileDirError
from IPython.paths import get_ipython_dir, get_ipython_package_dir
from IPython.utils.path import ensure_dir_exists
from traitlets import (
class BaseIPythonApplication(Application):
    name = 'ipython'
    description = 'IPython: an enhanced interactive Python shell.'
    version = Unicode(release.version)
    aliases = base_aliases
    flags = base_flags
    classes = List([ProfileDir])
    python_config_loader_class = ProfileAwareConfigLoader
    config_file_specified = Set()
    config_file_name = Unicode()

    @default('config_file_name')
    def _config_file_name_default(self):
        return self.name.replace('-', '_') + u'_config.py'

    @observe('config_file_name')
    def _config_file_name_changed(self, change):
        if change['new'] != change['old']:
            self.config_file_specified.add(change['new'])
    builtin_profile_dir = Unicode(os.path.join(get_ipython_package_dir(), u'config', u'profile', u'default'))
    config_file_paths = List(Unicode())

    @default('config_file_paths')
    def _config_file_paths_default(self):
        return []
    extra_config_file = Unicode(help='Path to an extra config file to load.\n    \n    If specified, load this config file in addition to any other IPython config.\n    ').tag(config=True)

    @observe('extra_config_file')
    def _extra_config_file_changed(self, change):
        old = change['old']
        new = change['new']
        try:
            self.config_files.remove(old)
        except ValueError:
            pass
        self.config_file_specified.add(new)
        self.config_files.append(new)
    profile = Unicode(u'default', help='The IPython profile to use.').tag(config=True)

    @observe('profile')
    def _profile_changed(self, change):
        self.builtin_profile_dir = os.path.join(get_ipython_package_dir(), u'config', u'profile', change['new'])
    add_ipython_dir_to_sys_path = Bool(False, 'Should the IPython profile directory be added to sys path ?\n\n        This option was non-existing before IPython 8.0, and ipython_dir was added to\n        sys path to allow import of extensions present there. This was historical\n        baggage from when pip did not exist. This now default to false,\n        but can be set to true for legacy reasons.\n        ').tag(config=True)
    ipython_dir = Unicode(help='\n        The name of the IPython directory. This directory is used for logging\n        configuration (through profiles), history storage, etc. The default\n        is usually $HOME/.ipython. This option can also be specified through\n        the environment variable IPYTHONDIR.\n        ').tag(config=True)

    @default('ipython_dir')
    def _ipython_dir_default(self):
        d = get_ipython_dir()
        self._ipython_dir_changed({'name': 'ipython_dir', 'old': d, 'new': d})
        return d
    _in_init_profile_dir = False
    profile_dir = Instance(ProfileDir, allow_none=True)

    @default('profile_dir')
    def _profile_dir_default(self):
        if self._in_init_profile_dir:
            return
        self.init_profile_dir()
        return self.profile_dir
    overwrite = Bool(False, help='Whether to overwrite existing config files when copying').tag(config=True)
    auto_create = Bool(False, help="Whether to create profile dir if it doesn't exist").tag(config=True)
    config_files = List(Unicode())

    @default('config_files')
    def _config_files_default(self):
        return [self.config_file_name]
    copy_config_files = Bool(False, help='Whether to install the default config files into the profile dir.\n        If a new profile is being created, and IPython contains config files for that\n        profile, then they will be staged into the new directory.  Otherwise,\n        default config files will be automatically generated.\n        ').tag(config=True)
    verbose_crash = Bool(False, help='Create a massive crash report when IPython encounters what may be an\n        internal error.  The default is to append a short message to the\n        usual traceback').tag(config=True)
    crash_handler_class = Type(crashhandler.CrashHandler)

    @catch_config_error
    def __init__(self, **kwargs):
        super(BaseIPythonApplication, self).__init__(**kwargs)
        try:
            os.getcwd()
        except:
            self.log.error("Current working directory doesn't exist.")
            self.exit(1)

    def init_crash_handler(self):
        """Create a crash handler, typically setting sys.excepthook to it."""
        self.crash_handler = self.crash_handler_class(self)
        sys.excepthook = self.excepthook

        def unset_crashhandler():
            sys.excepthook = sys.__excepthook__
        atexit.register(unset_crashhandler)

    def excepthook(self, etype, evalue, tb):
        """this is sys.excepthook after init_crashhandler

        set self.verbose_crash=True to use our full crashhandler, instead of
        a regular traceback with a short message (crash_handler_lite)
        """
        if self.verbose_crash:
            return self.crash_handler(etype, evalue, tb)
        else:
            return crashhandler.crash_handler_lite(etype, evalue, tb)

    @observe('ipython_dir')
    def _ipython_dir_changed(self, change):
        old = change['old']
        new = change['new']
        if old is not Undefined:
            str_old = os.path.abspath(old)
            if str_old in sys.path:
                sys.path.remove(str_old)
        if self.add_ipython_dir_to_sys_path:
            str_path = os.path.abspath(new)
            sys.path.append(str_path)
            ensure_dir_exists(new)
            readme = os.path.join(new, 'README')
            readme_src = os.path.join(get_ipython_package_dir(), 'config', 'profile', 'README')
            if not os.path.exists(readme) and os.path.exists(readme_src):
                shutil.copy(readme_src, readme)
            for d in ('extensions', 'nbextensions'):
                path = os.path.join(new, d)
                try:
                    ensure_dir_exists(path)
                except OSError as e:
                    self.log.error("couldn't create path %s: %s", path, e)
            self.log.debug('IPYTHONDIR set to: %s', new)

    def load_config_file(self, suppress_errors=IPYTHON_SUPPRESS_CONFIG_ERRORS):
        """Load the config file.

        By default, errors in loading config are handled, and a warning
        printed on screen. For testing, the suppress_errors option is set
        to False, so errors will make tests fail.

        `suppress_errors` default value is to be `None` in which case the
        behavior default to the one of `traitlets.Application`.

        The default value can be set :
           - to `False` by setting 'IPYTHON_SUPPRESS_CONFIG_ERRORS' environment variable to '0', or 'false' (case insensitive).
           - to `True` by setting 'IPYTHON_SUPPRESS_CONFIG_ERRORS' environment variable to '1' or 'true' (case insensitive).
           - to `None` by setting 'IPYTHON_SUPPRESS_CONFIG_ERRORS' environment variable to '' (empty string) or leaving it unset.

        Any other value are invalid, and will make IPython exit with a non-zero return code.
        """
        self.log.debug('Searching path %s for config files', self.config_file_paths)
        base_config = 'ipython_config.py'
        self.log.debug('Attempting to load config file: %s' % base_config)
        try:
            if suppress_errors is not None:
                old_value = Application.raise_config_file_errors
                Application.raise_config_file_errors = not suppress_errors
            Application.load_config_file(self, base_config, path=self.config_file_paths)
        except ConfigFileNotFound:
            self.log.debug('Config file %s not found', base_config)
            pass
        if suppress_errors is not None:
            Application.raise_config_file_errors = old_value
        for config_file_name in self.config_files:
            if not config_file_name or config_file_name == base_config:
                continue
            self.log.debug('Attempting to load config file: %s' % self.config_file_name)
            try:
                Application.load_config_file(self, config_file_name, path=self.config_file_paths)
            except ConfigFileNotFound:
                if config_file_name in self.config_file_specified:
                    msg = self.log.warning
                else:
                    msg = self.log.debug
                msg('Config file not found, skipping: %s', config_file_name)
            except Exception:
                if not suppress_errors:
                    raise
                self.log.warning('Error loading config file: %s' % self.config_file_name, exc_info=True)

    def init_profile_dir(self):
        """initialize the profile dir"""
        self._in_init_profile_dir = True
        if self.profile_dir is not None:
            return
        if 'ProfileDir.location' not in self.config:
            try:
                p = ProfileDir.find_profile_dir_by_name(self.ipython_dir, self.profile, self.config)
            except ProfileDirError:
                if self.auto_create or self.profile == 'default':
                    try:
                        p = ProfileDir.create_profile_dir_by_name(self.ipython_dir, self.profile, self.config)
                    except ProfileDirError:
                        self.log.fatal('Could not create profile: %r' % self.profile)
                        self.exit(1)
                    else:
                        self.log.info('Created profile dir: %r' % p.location)
                else:
                    self.log.fatal('Profile %r not found.' % self.profile)
                    self.exit(1)
            else:
                self.log.debug('Using existing profile dir: %r', p.location)
        else:
            location = self.config.ProfileDir.location
            try:
                p = ProfileDir.find_profile_dir(location, self.config)
            except ProfileDirError:
                if self.auto_create:
                    try:
                        p = ProfileDir.create_profile_dir(location, self.config)
                    except ProfileDirError:
                        self.log.fatal('Could not create profile directory: %r' % location)
                        self.exit(1)
                    else:
                        self.log.debug('Creating new profile dir: %r' % location)
                else:
                    self.log.fatal('Profile directory %r not found.' % location)
                    self.exit(1)
            else:
                self.log.debug('Using existing profile dir: %r', p.location)
            dir_name = os.path.basename(p.location)
            if dir_name.startswith('profile_'):
                self.profile = dir_name[8:]
        self.profile_dir = p
        self.config_file_paths.append(p.location)
        self._in_init_profile_dir = False

    def init_config_files(self):
        """[optionally] copy default config files into profile dir."""
        self.config_file_paths.extend(ENV_CONFIG_DIRS)
        self.config_file_paths.extend(SYSTEM_CONFIG_DIRS)
        path = Path(self.builtin_profile_dir)
        if self.copy_config_files:
            src = self.profile
            cfg = self.config_file_name
            if path and (path / cfg).exists():
                self.log.warning('Staging %r from %s into %r [overwrite=%s]' % (cfg, src, self.profile_dir.location, self.overwrite))
                self.profile_dir.copy_config_file(cfg, path=path, overwrite=self.overwrite)
            else:
                self.stage_default_config_file()
        else:
            files = path.glob('*.py')
            for fullpath in files:
                cfg = fullpath.name
                if self.profile_dir.copy_config_file(cfg, path=path, overwrite=False):
                    self.log.warning('Staging bundled %s from %s into %r' % (cfg, self.profile, self.profile_dir.location))

    def stage_default_config_file(self):
        """auto generate default config file, and stage it into the profile."""
        s = self.generate_config_file()
        config_file = Path(self.profile_dir.location) / self.config_file_name
        if self.overwrite or not config_file.exists():
            self.log.warning('Generating default config file: %r', config_file)
            config_file.write_text(s, encoding='utf-8')

    @catch_config_error
    def initialize(self, argv=None):
        self.parse_command_line(argv)
        self.init_crash_handler()
        if self.subapp is not None:
            return
        cl_config = deepcopy(self.config)
        self.init_profile_dir()
        self.init_config_files()
        self.load_config_file()
        self.update_config(cl_config)