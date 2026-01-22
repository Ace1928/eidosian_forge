from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import boto
from boto import config
from gslib import context_config
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import CloudApi
from gslib.cs_api_map import ApiMapConstants
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.utils import boto_util
def _LoadApi(self, provider, api_selector):
    """Loads a CloudApi into the loaded_apis map for this class.

    Args:
      provider: Provider to load the API for.
      api_selector: cs_api_map.ApiSelector defining the API type.
    """
    if provider not in self.api_map[ApiMapConstants.API_MAP]:
        raise ArgumentException('gsutil Cloud API map contains no entry for provider %s.' % provider)
    if api_selector not in self.api_map[ApiMapConstants.API_MAP][provider]:
        raise ArgumentException('gsutil Cloud API map does not support API %s for provider %s.' % (api_selector, provider))
    self.loaded_apis[provider][api_selector] = self.api_map[ApiMapConstants.API_MAP][provider][api_selector](self.bucket_storage_uri_class, self.logger, self.status_queue, provider=provider, debug=self.debug, http_headers=self.http_headers, trace_token=self.trace_token, perf_trace_token=self.perf_trace_token, user_project=self.user_project)