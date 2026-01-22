from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import ipaddress
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
import six
def FallthroughToLocationProperty(location_refs, flag_name, failure_msg):
    """Provides a list containing composer/location if `location_refs` is empty.

  This intended to be used as a fallthrough for a plural Location resource arg.
  The built-in fallthrough for plural resource args doesn't play well with
  properties, as it will iterate over each character in the string and parse
  it as the resource type. This function will parse the entire property and
  return a singleton list if `location_refs` is empty.

  Args:
    location_refs: [core.resources.Resource], a possibly empty list of location
      resource references
    flag_name: str, if `location_refs` is empty, and the composer/location
      property is also missing, an error message will be reported that will
      advise the user to set this flag name
    failure_msg: str, an error message to accompany the advisory described in
      the docs for `flag_name`.

  Returns:
    [core.resources.Resource]: a non-empty list of location resourc references.
    If `location_refs` was non-empty, it will be the same list, otherwise it
    will be a singleton list containing the value of the [composer/location]
    property.

  Raises:
    exceptions.RequiredArgumentException: both the user-provided locations
        and property fallback were empty
  """
    if location_refs:
        return location_refs
    fallthrough_location = parsers.GetLocation(required=False)
    if fallthrough_location:
        return [parsers.ParseLocation(fallthrough_location)]
    else:
        raise exceptions.RequiredArgumentException(flag_name, failure_msg)