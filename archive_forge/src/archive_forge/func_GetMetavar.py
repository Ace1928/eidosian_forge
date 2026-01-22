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
def GetMetavar(specified_metavar, flag_type, flag_name):
    """Gets the metavar for specific flag.

  Args:
    specified_metavar: str, metavar that is specified by user.
    flag_type: (str)->None, type function of the flag.
    flag_name: str, name of the flag

  Returns:
    str | None, the flag's metavar
  """
    if specified_metavar:
        metavar = specified_metavar
    elif isinstance(flag_type, arg_parsers.ArgDict):
        metavar = None
    elif isinstance(flag_type, arg_parsers.ArgList):
        metavar = flag_name
    else:
        metavar = None
    if metavar:
        return resource_property.ConvertToAngrySnakeCase(metavar.replace('-', '_'))
    else:
        return None