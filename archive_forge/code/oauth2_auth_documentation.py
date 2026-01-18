from oauthlib.oauth2 import WebApplicationClient, InsecureTransportError
from oauthlib.oauth2 import is_secure_transport
from requests.auth import AuthBase
Append an OAuth 2 token to the request.

        Note that currently HTTPS is required for all requests. There may be
        a token type that allows for plain HTTP in the future and then this
        should be updated to allow plain HTTP on a white list basis.
        