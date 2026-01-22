import abc
from keystoneauth1.identity.v3 import base
from keystoneauth1.identity.v3 import token
class FederationBaseAuth(_Rescoped):
    """Federation authentication plugin.

    :param auth_url: URL of the Identity Service
    :type auth_url: string
    :param identity_provider: name of the Identity Provider the client
                              will authenticate against. This parameter
                              will be used to build a dynamic URL used to
                              obtain unscoped OpenStack token.
    :type identity_provider: string
    :param protocol: name of the protocol the client will authenticate
                     against.
    :type protocol: string

    """

    def __init__(self, auth_url, identity_provider, protocol, **kwargs):
        super(FederationBaseAuth, self).__init__(auth_url=auth_url, **kwargs)
        self.identity_provider = identity_provider
        self.protocol = protocol

    @property
    def federated_token_url(self):
        """Full URL where authorization data is sent."""
        host = self.auth_url.rstrip('/')
        if not host.endswith('v3'):
            host += '/v3'
        values = {'host': host, 'identity_provider': self.identity_provider, 'protocol': self.protocol}
        url = '%(host)s/OS-FEDERATION/identity_providers/%(identity_provider)s/protocols/%(protocol)s/auth'
        url = url % values
        return url