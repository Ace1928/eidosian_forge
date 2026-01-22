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
def GetAttributeFlags(arg_data, arg_name, resource_collection, shared_resource_args):
    """Gets a list of attribute flags for the given resource arg.

  Args:
    arg_data: yaml_arg_schema.YAMLResourceArgument, data used to generate the
      resource argument
    arg_name: str, name of the anchor resource arg
    resource_collection: registry.APICollection | None, collection used to
      create resource argument.
    shared_resource_args: [str], list of resource args to ignore

  Returns:
    A list of base.Argument resource attribute flags.
  """
    name = GetFlagName(arg_name)
    resource_arg = arg_data.GenerateResourceArg(resource_collection, name, shared_resource_args).GetInfo(name)
    return resource_arg.GetAttributeArgs()[:-1]