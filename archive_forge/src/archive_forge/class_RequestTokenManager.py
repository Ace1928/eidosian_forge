import urllib.parse as urlparse
from keystoneauth1 import plugin
from keystoneclient import base
from keystoneclient.v3.contrib.oauth1 import utils
class RequestTokenManager(base.CrudManager):
    """Manager class for manipulating identity OAuth request tokens."""
    resource_class = RequestToken

    def authorize(self, request_token, roles):
        """Authorize a request token with specific roles.

        Utilize Identity API operation:
        PUT /OS-OAUTH1/authorize/$request_token_id

        :param request_token: a request token that will be authorized, and
            can be exchanged for an access token.
        :param roles: a list of roles, that will be delegated to the user.
        """
        request_id = urlparse.quote(base.getid(request_token))
        endpoint = utils.OAUTH_PATH + '/authorize/%s' % request_id
        body = {'roles': [{'id': base.getid(r_id)} for r_id in roles]}
        return self._put(endpoint, body, 'token')

    def create(self, consumer_key, consumer_secret, project):
        endpoint = utils.OAUTH_PATH + '/request_token'
        headers = {'requested-project-id': base.getid(project)}
        oauth_client = oauth1.Client(consumer_key, client_secret=consumer_secret, signature_method=oauth1.SIGNATURE_HMAC, callback_uri='oob')
        url = self.client.get_endpoint(interface=plugin.AUTH_INTERFACE).rstrip('/')
        url, headers, body = oauth_client.sign(url + endpoint, http_method='POST', headers=headers)
        resp, body = self.client.post(endpoint, headers=headers)
        token = utils.get_oauth_token_from_body(resp.content)
        return self._prepare_return_value(resp, self.resource_class(self, token))