from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.api_gateway import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.api_gateway import common_flags
Returns an API Config object.

    Args:
      api_config_ref: A parsed resource reference for the API.
      view: Optional string. If specified as FULL, the source config files will
        be returned.

    Returns:
      An API Config object.

    Raises:
      calliope.InvalidArgumentException: If an invalid view (i.e. not FULL,
      BASIC, or none) was
      provided.
    