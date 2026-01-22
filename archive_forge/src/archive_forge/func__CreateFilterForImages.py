from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.services import enable_api
import six
def _CreateFilterForImages(prefixes, custom_filter, images):
    """Creates a list of filters from a docker image prefix, a custom filter and fully-qualified image URLs.

  Args:
    prefixes: URL prefixes. Only metadata of images with any of these prefixes
      will be retrieved.
    custom_filter: user provided filter string.
    images: fully-qualified docker image URLs. Only metadata of these images
      will be retrieved.

  Returns:
    A filter string to send to the containeranalysis API.
  """
    occ_filter = filter_util.ContainerAnalysisFilter()
    occ_filter.WithResourcePrefixes(prefixes)
    occ_filter.WithResources(images)
    occ_filter.WithCustomFilter(custom_filter)
    return occ_filter.GetChunkifiedFilters()