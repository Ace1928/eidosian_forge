import requests.auth
from keystoneauth1.exceptions import ClientException
from keystoneauth1.identity.v3 import base
Fetch authentication headers for message.

        :param session: The session object that the auth_plugin belongs to.
        :type session: keystoneauth1.session.Session

        :returns: Headers that are set to authenticate a message or None for
                  failure. Note that when checking this value that the empty
                  dict is a valid, non-failure response.
        :rtype: dict
        