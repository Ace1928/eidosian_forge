from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.backend_buckets import (
from googlecloudsdk.command_lib.compute.backend_services import (
from googlecloudsdk.command_lib.compute.url_maps import flags
from googlecloudsdk.command_lib.compute.url_maps import url_maps_utils
from googlecloudsdk.core import properties
import six
def _GetBackendServiceParamsForUrlMap(url_map, url_map_ref):
    params = {'project': properties.VALUES.core.project.GetOrFail}
    if hasattr(url_map, 'region') and url_map.region:
        params['region'] = url_map_ref.region
    return params