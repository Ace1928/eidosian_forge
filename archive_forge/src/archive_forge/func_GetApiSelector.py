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
def GetApiSelector(self, provider=None):
    """Returns a cs_api_map.ApiSelector based on input and configuration.

    Args:
      provider: Provider to return the ApiSelector for.  If None, class-wide
                default is used.

    Returns:
      cs_api_map.ApiSelector that will be used for calls to the delegator
      for this provider.
    """
    selected_provider = provider or self.provider
    if not selected_provider:
        raise ArgumentException('No provider selected for CloudApi')
    if selected_provider not in self.api_map[ApiMapConstants.DEFAULT_MAP] or self.api_map[ApiMapConstants.DEFAULT_MAP][selected_provider] not in self.api_map[ApiMapConstants.API_MAP][selected_provider]:
        raise ArgumentException('No default api available for provider %s' % selected_provider)
    if selected_provider not in self.api_map[ApiMapConstants.SUPPORT_MAP]:
        raise ArgumentException('No supported apis available for provider %s' % selected_provider)
    api = self.api_map[ApiMapConstants.DEFAULT_MAP][selected_provider]
    using_gs_hmac = provider == 'gs' and boto_util.UsingGsHmac()
    configured_encryption = provider == 'gs' and (config.has_option('GSUtil', 'encryption_key') or config.has_option('GSUtil', 'decryption_key1'))
    if using_gs_hmac and configured_encryption:
        raise CommandException('gsutil does not support HMAC credentials with customer-supplied encryption keys (CSEK) or customer-managed KMS encryption keys (CMEK). Please generate and include non-HMAC credentials in your .boto configuration file, or to access public encrypted objects, remove your HMAC credentials.')
    elif using_gs_hmac:
        api = ApiSelector.XML
    elif configured_encryption:
        api = ApiSelector.JSON
    elif self.prefer_api in self.api_map[ApiMapConstants.SUPPORT_MAP][selected_provider]:
        api = self.prefer_api
    if api == ApiSelector.XML and context_config.get_context_config() and context_config.get_context_config().use_client_certificate:
        raise ArgumentException('User enabled mTLS by setting "use_client_certificate", but mTLS is not supported for the selected XML API. Try configuring for  the GCS JSON API or setting "use_client_certificate" to "False" in the Boto config.')
    return api