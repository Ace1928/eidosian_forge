from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import unicode_literals
import abc
from collections.abc import Callable
import dataclasses
from typing import Any
from apitools.base.protorpclite import messages as apitools_messages
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import module_util
def FormatResourceAttrStr(format_string, resource_ref, display_name=None, display_resource_type=None):
    """Formats a string with all the attributes of the given resource ref.

  Args:
    format_string: str, The format string.
    resource_ref: resources.Resource, The resource reference to extract
      attributes from.
    display_name: the display name for the resource.
    display_resource_type:

  Returns:
    str, The formatted string.
  """
    if resource_ref:
        d = resource_ref.AsDict()
        d[NAME_FORMAT_KEY] = display_name or resource_ref.Name()
        d[RESOURCE_ID_FORMAT_KEY] = resource_ref.Name()
        d[REL_NAME_FORMAT_KEY] = resource_ref.RelativeName()
    else:
        d = {NAME_FORMAT_KEY: display_name}
    d[RESOURCE_TYPE_FORMAT_KEY] = display_resource_type
    try:
        return format_string.format(**d)
    except KeyError as err:
        if err.args:
            raise KeyError('Key [{}] does not exist. Must specify one of the following keys instead: {}'.format(err.args[0], ', '.join(d.keys())))
        else:
            raise err