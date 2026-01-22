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
def GetChildFieldName(api_field):
    """Gets the child field name from the api field.

  If api field path is multiple levels deep, return the last field name.
  i.e. 'x.y.z' would return 'z'

  Args:
    api_field: str, full api field path

  Returns:
    str, child api field
  """
    return api_field.rpartition('.')[-1]