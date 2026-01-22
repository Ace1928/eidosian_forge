from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import tempfile
from apitools.base.protorpclite.messages import DecodeError
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from apitools.base.py import transfer
from googlecloudsdk.api_lib.genomics import exceptions as genomics_exceptions
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
import six
def GetQueryFields(referenced_fields, prefix):
    """Returns the comma separated list of field names referenced by the command.

  Args:
    referenced_fields: A list of field names referenced by the format and filter
      expressions.
    prefix: The referenced field name resource prefix.

  Returns:
    The comma separated list of field names referenced by the command.
  """
    if not referenced_fields:
        return None
    return ','.join(['nextPageToken'] + ['.'.join([prefix, field]) for field in referenced_fields])