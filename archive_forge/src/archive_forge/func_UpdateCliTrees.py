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
def UpdateCliTrees(cli=None, commands=None, directory=None, tarball=None, force=False, verbose=False, warn_on_exceptions=False):
    """(re)generates the CLI trees in directory if non-existent or out of date.

  This function uses the progress tracker because some of the updates can
  take ~minutes.

  Args:
    cli: The default CLI. If not None then the default CLI is also updated.
    commands: Update only the commands in this list.
    directory: The directory containing the CLI tree JSON files. If None
      then the default installation directories are used.
    tarball: For packaging CLI trees. --commands specifies one command that is
      a relative path in this tarball. The tarball is extracted to a temporary
      directory and the command path is adjusted to point to the temporary
      directory.
    force: Update all exitsing trees by forcing them to be out of date if True.
    verbose: Display a status line for up to date CLI trees if True.
    warn_on_exceptions: Emits warning messages in lieu of exceptions. Used
      during installation.

  Raises:
    CliTreeGenerationError: CLI tree generation failed for a command in
      commands.
  """
    directories = _GetDirectories(directory=directory, warn_on_exceptions=warn_on_exceptions)
    if not commands:
        commands = set([cli_tree.DEFAULT_CLI_NAME] + list(GENERATORS.keys()))
    failed = []
    for command in sorted(commands):
        if command != cli_tree.DEFAULT_CLI_NAME:
            tree = LoadOrGenerate(command, directories=directories, tarball=tarball, force=force, verbose=verbose, warn_on_exceptions=warn_on_exceptions)
            if not tree:
                failed.append(command)
        elif cli:

            def _Mtime(path):
                try:
                    return os.path.getmtime(path)
                except OSError:
                    return 0
            cli_tree_path = cli_tree.CliTreeConfigPath(directory=directories[-1])
            cli_tree.Load(cli=cli, path=cli_tree_path, force=force, verbose=verbose)
            completion_tree_path = lookup.CompletionCliTreePath(directory=directories[0])
            cli_tree_mtime = _Mtime(cli_tree_path)
            completion_tree_mtime = _Mtime(completion_tree_path)
            if force or not completion_tree_mtime or completion_tree_mtime < cli_tree_mtime:
                files.MakeDir(os.path.dirname(completion_tree_path))
                with files.FileWriter(completion_tree_path) as f:
                    generate_static.ListCompletionTree(cli, out=f)
            elif verbose:
                log.status.Print('[{}] static completion CLI tree is up to date.'.format(command))
    if failed:
        message = 'CLI tree generation failed for [{}].'.format(', '.join(sorted(failed)))
        if not warn_on_exceptions:
            raise CliTreeGenerationError(message)
        log.warning(message)