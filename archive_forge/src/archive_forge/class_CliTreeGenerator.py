from a man-ish style runtime document.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import os
import re
import shlex
import subprocess
import tarfile
import textwrap
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.static_completion import generate as generate_static
from googlecloudsdk.command_lib.static_completion import lookup
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
from six.moves import range
class CliTreeGenerator(six.with_metaclass(abc.ABCMeta, object)):
    """Base CLI tree generator.

  Attributes:
    command_name: str, The name of the CLI tree command.
  """
    _FAILURES = None

    @classmethod
    def MemoizeFailures(cls, enable):
        """Memoizes failed attempts and doesn't repeat them if enable is True."""
        cls._FAILURES = set() if enable else None

    @classmethod
    def AlreadyFailed(cls, command):
        """Returns True if man page request for command already failed."""
        return command in cls._FAILURES if cls._FAILURES else False

    @classmethod
    def AddFailure(cls, command):
        """Add command to the set of failed man generations."""
        if cls._FAILURES is not None:
            cls._FAILURES.add(command)

    def __init__(self, command_name, root_command_args=None):
        """Initializes the CLI tree generator.

    Args:
      command_name: str, The name of the CLI tree command (e.g. 'gsutil').
      root_command_args: [str], The argument list to invoke the root CLI tree
        command. Examples:
        * ['gcloud']
        * ['python', '/tmp/tarball_dir/gsutil/gsutil']
    Raises:
      CommandInvocationError: If the provided root command cannot be invoked.
    """
        if root_command_args:
            with files.FileWriter(os.devnull) as devnull:
                try:
                    subprocess.Popen(root_command_args, stdin=devnull, stdout=devnull, stderr=devnull).communicate()
                except OSError as e:
                    raise CommandInvocationError(e)
        self.command_name = command_name
        self._root_command_args = root_command_args or [command_name]
        self._cli_version = None

    def Run(self, cmd):
        """Runs the root command with args given by cmd and returns the output.

    Args:
      cmd: [str], List of arguments to the root command.
    Returns:
      str, Output of the given command.
    """
        return encoding.Decode(subprocess.check_output(self._root_command_args + cmd))

    def GetVersion(self):
        """Returns the CLI_VERSION string."""
        if not self._cli_version:
            try:
                self._cli_version = self.Run(['version']).split()[-1]
            except:
                self._cli_version = cli_tree.CLI_VERSION_UNKNOWN
        return self._cli_version

    @abc.abstractmethod
    def Generate(self):
        """Generates and returns the CLI tree dict."""
        return None

    def FindTreeFile(self, directories):
        """Returns (path,f) open for read for the first CLI tree in directories."""
        for directory in directories or _GetDirectories(warn_on_exceptions=True):
            path = os.path.join(directory or '.', self.command_name) + '.json'
            try:
                return (path, files.FileReader(path))
            except files.Error:
                pass
        return (path, None)

    def IsUpToDate(self, tree, verbose=False):
        """Returns a bool tuple (readonly, up_to_date)."""
        actual_cli_version = tree.get(cli_tree.LOOKUP_CLI_VERSION)
        readonly = actual_cli_version == cli_tree.CLI_VERSION_READONLY
        actual_tree_version = tree.get(cli_tree.LOOKUP_VERSION)
        if actual_tree_version != cli_tree.VERSION:
            return (readonly, False)
        expected_cli_version = self.GetVersion()
        if readonly:
            pass
        elif expected_cli_version == cli_tree.CLI_VERSION_UNKNOWN:
            pass
        elif actual_cli_version != expected_cli_version:
            return (readonly, False)
        if verbose:
            log.status.Print('[{}] CLI tree version [{}] is up to date.'.format(self.command_name, actual_cli_version))
        return (readonly, True)

    def LoadOrGenerate(self, directories=None, force=False, generate=True, ignore_out_of_date=False, tarball=False, verbose=False, warn_on_exceptions=False):
        """Loads the CLI tree or generates it if necessary, and returns the tree."""
        f = None
        try:
            path, f = self.FindTreeFile(directories)
            if f:
                up_to_date = False
                try:
                    tree = json.load(f)
                except ValueError:
                    tree = None
                if tree:
                    readonly, up_to_date = self.IsUpToDate(tree, verbose=verbose)
                    if readonly:
                        return tree
                    elif up_to_date:
                        if not force:
                            return tree
                    elif ignore_out_of_date:
                        return None
        finally:
            if f:
                f.close()

        def _Generate():
            """Helper that generates a CLI tree and writes it to a JSON file."""
            tree = self.Generate()
            if tree:
                try:
                    f = files.FileWriter(path)
                except files.Error as e:
                    directory, _ = os.path.split(path)
                    try:
                        files.MakeDir(directory)
                        f = files.FileWriter(path)
                    except files.Error:
                        if not warn_on_exceptions:
                            raise
                        log.warning(six.text_type(e))
                        return None
                with f:
                    resource_printer.Print(tree, print_format='json', out=f)
            return tree
        if not generate:
            raise NoCliTreeForCommandError('No CLI tree for [{}].'.format(self.command_name))
        if not verbose:
            return _Generate()
        with progress_tracker.ProgressTracker('{} the [{}] CLI tree'.format('Updating' if f else 'Generating', self.command_name)):
            return _Generate()