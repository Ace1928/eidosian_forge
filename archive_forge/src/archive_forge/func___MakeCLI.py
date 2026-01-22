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