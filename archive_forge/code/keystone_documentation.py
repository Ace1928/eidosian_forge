import urllib.parse
from keystoneauth1 import discover
from keystoneauth1 import exceptions as ka_exc
from keystoneauth1.identity import v2 as v2_auth
from keystoneauth1.identity import v3 as v3_auth
from keystoneauth1 import session
from zaqarclient.auth import base
from zaqarclient import errors
Get an authtenticated client using credentials in the keyword args.

        :param api_version: the API version to use ('1' or '2')
        :param request: The request spec instance to modify with
            the auth information.
        