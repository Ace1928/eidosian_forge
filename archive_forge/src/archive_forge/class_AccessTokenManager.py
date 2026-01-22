from keystoneauth1 import plugin
from keystoneclient import base
from keystoneclient.v3.contrib.oauth1 import utils
class AccessTokenManager(base.CrudManager):
    """Manager class for manipulating identity OAuth access tokens."""
    resource_class = AccessToken

    def create(self, consumer_key, consumer_secret, request_key, request_secret, verifier):
        endpoint = utils.OAUTH_PATH + '/access_token'
        oauth_client = oauth1.Client(consumer_key, client_secret=consumer_secret, resource_owner_key=request_key, resource_owner_secret=request_secret, signature_method=oauth1.SIGNATURE_HMAC, verifier=verifier)
        url = self.client.get_endpoint(interface=plugin.AUTH_INTERFACE).rstrip('/')
        url, headers, body = oauth_client.sign(url + endpoint, http_method='POST')
        resp, body = self.client.post(endpoint, headers=headers)
        token = utils.get_oauth_token_from_body(resp.content)
        return self._prepare_return_value(resp, self.resource_class(self, token))