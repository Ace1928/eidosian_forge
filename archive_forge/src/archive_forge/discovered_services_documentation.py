from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.apphub import utils as api_lib_utils
from googlecloudsdk.calliope import base
Lookup a discovered service in the Project/location with a given uri.

    Args:
      parent: str, projects/{projectId_or_projectNumber}/locations/{location}
      uri: str, GCP resource URI to find service for Accepts both project number
        and project id and does translation when needed.

    Returns:
       discoveredService: Discovered service
    