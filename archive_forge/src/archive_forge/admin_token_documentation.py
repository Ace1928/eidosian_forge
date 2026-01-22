from keystoneauth1 import loading
from keystoneauth1 import token_endpoint
Use an existing token and a known endpoint to perform requests.

    This plugin is primarily useful for development or for use with identity
    service ADMIN tokens. Because this token is used directly there is no
    fetching a service catalog or determining scope information and so it
    cannot be used by clients that expect use this scope information.

    Because there is no service catalog the endpoint that is supplied with
    initialization is used for all operations performed with this plugin so
    must be the full base URL to an actual service.
    