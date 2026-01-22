from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.firebase.test import arg_validate
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
import six
def _SplitArgFileAndGroup(file_and_group_str):
    """Parses an ARGSPEC and returns the arg filename and arg group name."""
    index = file_and_group_str.rfind(':')
    if index < 0 or (index == 2 and file_and_group_str.startswith('gs://')):
        raise exceptions.InvalidArgException('arg-spec', 'Format must be ARG_FILE:ARG_GROUP_NAME')
    return (file_and_group_str[:index], file_and_group_str[index + 1:])