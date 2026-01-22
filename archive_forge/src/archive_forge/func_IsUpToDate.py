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