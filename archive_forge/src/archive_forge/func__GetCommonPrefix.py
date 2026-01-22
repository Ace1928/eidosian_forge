from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import re
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import http_encoding
import six
def _GetCommonPrefix(longest_arr, arr):
    """Gets the long common sub list between two lists."""
    new_arr = []
    for i, longest_substr_seg in enumerate(longest_arr):
        if i >= len(arr) or arr[i] != longest_substr_seg:
            break
        new_arr.append(arr[i])
    return new_arr