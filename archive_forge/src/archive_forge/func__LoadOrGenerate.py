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
def _LoadOrGenerate(command_path, command_name):
    """Helper."""
    if command_name in GENERATORS:
        command_args = command_executable_args + [command_path]
        try:
            generator = GENERATORS[command_name](command_name, root_command_args=command_args)
        except CommandInvocationError as e:
            if verbose:
                log.status.Print('Command [{}] could not be invoked:\n{}'.format(command, e))
            return None
    else:
        generator = ManPageCliTreeGenerator(command_name)
    try:
        return generator.LoadOrGenerate(directories=directories, force=force, generate=generate, ignore_out_of_date=ignore_out_of_date, verbose=verbose, warn_on_exceptions=warn_on_exceptions)
    except NoManPageTextForCommandError:
        pass
    return None