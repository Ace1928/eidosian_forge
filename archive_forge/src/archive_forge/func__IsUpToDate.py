from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import re
import sys
import textwrap
from googlecloudsdk.calliope import walker
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import module_util
from googlecloudsdk.core.util import files
import six
def _IsUpToDate(tree, path, ignore_errors, verbose):
    """Returns True if the CLI tree on path is up to date.

  Args:
    tree: The loaded CLI tree.
    path: The path tree was loaded from.
    ignore_errors: If True then return True if tree versions match. Otherwise
      raise exceptions on version mismatch.
    verbose: Display a status line for up to date CLI trees if True.

  Raises:
    CliTreeVersionError: tree version mismatch.
    CliCommandVersionError: CLI command version mismatch.

  Returns:
    True if tree versions match.
  """
    expected_tree_version = VERSION
    actual_tree_version = tree.get(LOOKUP_VERSION)
    if actual_tree_version != expected_tree_version:
        if not ignore_errors:
            raise CliCommandVersionError('CLI tree [{}] version is [{}], expected [{}]'.format(path, actual_tree_version, expected_tree_version))
        return False
    expected_command_version = _GetDefaultCliCommandVersion()
    actual_command_version = tree.get(LOOKUP_CLI_VERSION)
    test_versions = (TEST_CLI_VERSION_HEAD, TEST_CLI_VERSION_TEST)
    if actual_command_version in test_versions or expected_command_version in test_versions:
        pass
    elif actual_command_version != expected_command_version:
        if not ignore_errors:
            raise CliCommandVersionError('CLI tree [{}] command version is [{}], expected [{}]'.format(path, actual_command_version, expected_command_version))
        return False
    if verbose:
        from googlecloudsdk.core import log
        log.status.Print('[{}] CLI tree version [{}] is up to date.'.format(DEFAULT_CLI_NAME, expected_command_version))
    return True