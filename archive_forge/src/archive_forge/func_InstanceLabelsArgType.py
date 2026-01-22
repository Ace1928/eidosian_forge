from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib import redis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core.console import console_io
import six
def InstanceLabelsArgType(value):
    return arg_parsers.ArgDict(key_type=labels_util.KEY_FORMAT_VALIDATOR, value_type=labels_util.VALUE_FORMAT_VALIDATOR)(value)