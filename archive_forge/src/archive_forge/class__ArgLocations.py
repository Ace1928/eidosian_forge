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
class _ArgLocations(object):
    """--flags-file (arg,locations) info."""

    def __init__(self, arg, file_name, line_col, locations=None):
        self.arg = arg
        self.locations = locations.locations[:] if locations else []
        self.locations.append(_FlagLocation(file_name, line_col))

    def __str__(self):
        return ';'.join([six.text_type(location) for location in self.locations])

    def FileInStack(self, file_name):
        """Returns True if file_name is in the locations stack."""
        return any([file_name == x.file_name for x in self.locations])