from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
def GetSslPolicyForInsert(self, name, description, profile, min_tls_version, custom_features):
    """Returns the SslPolicy message for an insert request.

    Args:
      name: String representing the name of the SSL policy resource.
      description: String representing the description for the SSL policy
        resource.
      profile: String representing the SSL policy profile. Can be one of
        'COMPATIBLE', 'MODERN', 'RESTRICTED' or 'CUSTOM'.
      min_tls_version: String representing the minimum TLS version of the SSL
        policy. Can be one of 'TLS_1_0', 'TLS_1_1', 'TLS_1_2'.
      custom_features: The list of strings representing the custom features to
        use.

    Returns:
      The SslPolicy message object that can be used in an insert request.
    """
    return self._messages.SslPolicy(name=name, description=description, profile=self._messages.SslPolicy.ProfileValueValuesEnum(profile), minTlsVersion=self._messages.SslPolicy.MinTlsVersionValueValuesEnum(min_tls_version), customFeatures=custom_features)