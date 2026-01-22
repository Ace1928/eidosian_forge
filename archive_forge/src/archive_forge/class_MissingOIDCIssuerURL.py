from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class MissingOIDCIssuerURL(exceptions.Error):
    """Class for errors by missing OIDC issuer URL."""

    def __init__(self, config):
        message = 'Invalid OpenID Config: missing issuer: {}'.format(config)
        super(MissingOIDCIssuerURL, self).__init__(message)