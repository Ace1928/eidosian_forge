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
def _AddFlagsFileFlags(inject, flags_file, parent_locations=None):
    """Recursively append the flags file flags to inject."""
    flag = calliope_base.FLAGS_FILE_FLAG.name
    if parent_locations and parent_locations.FileInStack(flags_file):
        raise parser_errors.ArgumentError('{} recursive reference ({}).'.format(flag, parent_locations))
    if flags_file == '-':
        contents = sys.stdin.read()
    elif not os.path.exists(flags_file):
        raise parser_errors.ArgumentError('{} [{}] not found.'.format(flag, flags_file))
    else:
        contents = files.ReadFileContents(flags_file)
    if not contents:
        raise parser_errors.ArgumentError('{} [{}] is empty.'.format(flag, flags_file))
    data = yaml.load(contents, location_value=True)
    group = data if isinstance(data, list) else [data]
    for member in group:
        if not isinstance(member.value, dict):
            raise parser_errors.ArgumentError('{}:{}: {} file must contain a dictionary or list of dictionaries of flags.'.format(flags_file, member.lc.line + 1, flag))
        for arg, obj in six.iteritems(member.value):
            line_col = obj.lc
            value = yaml.strip_locations(obj)
            if arg == flag:
                file_list = obj.value if isinstance(obj.value, list) else [obj.value]
                for path in file_list:
                    locations = _ArgLocations(arg, flags_file, line_col, parent_locations)
                    _AddFlagsFileFlags(inject, path, locations)
                continue
            if isinstance(value, (type(None), bool)):
                separate_value_arg = False
            elif isinstance(value, (list, dict)):
                separate_value_arg = True
            else:
                separate_value_arg = False
                arg = '{}={}'.format(arg, value)
            inject.append(_FLAG_FILE_LINE_NAME)
            inject.append(_ArgLocations(arg, flags_file, line_col, parent_locations))
            inject.append(arg)
            if separate_value_arg:
                inject.append(value)