from keystoneauth1 import http_basic
from keystoneauth1 import loading
Use HTTP Basic authentication to perform requests.

    This can be used to instantiate clients for services deployed in
    standalone mode.

    There is no fetching a service catalog or determining scope information
    and so it cannot be used by clients that expect to use this scope
    information.

    