from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.url_maps import flags
from googlecloudsdk.command_lib.compute.url_maps import url_maps_utils
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_projector
def _GetUrlMapGetRequest(url_map_ref, client):
    if url_maps_utils.IsGlobalUrlMapRef(url_map_ref):
        return (client.apitools_client.urlMaps, 'Get', client.messages.ComputeUrlMapsGetRequest(project=properties.VALUES.core.project.GetOrFail(), urlMap=url_map_ref.Name()))
    else:
        return (client.apitools_client.regionUrlMaps, 'Get', client.messages.ComputeRegionUrlMapsGetRequest(project=properties.VALUES.core.project.GetOrFail(), urlMap=url_map_ref.Name(), region=url_map_ref.region))