from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import re
import textwrap
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def APIEndpoint():
    """Returns the current GKEHub API environment.

  Assumes prod endpoint if override is unset, unknown endpoint if overrides has
  unrecognized value.

  Returns:
    One of prod, staging, autopush, or unknown.
  """
    try:
        hub_endpoint_override = properties.VALUES.api_endpoint_overrides.Property('gkehub').Get()
    except properties.NoSuchPropertyError:
        hub_endpoint_override = None
    if not hub_endpoint_override or 'gkehub.googleapis.com' in hub_endpoint_override:
        return PROD_API
    elif 'staging-gkehub' in hub_endpoint_override:
        return STAGING_API
    elif 'autopush-gkehub' in hub_endpoint_override:
        return AUTOPUSH_API
    else:
        return UNKNOWN_API