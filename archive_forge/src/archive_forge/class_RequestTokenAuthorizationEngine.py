from __future__ import print_function
import httplib2
import json
import os
from select import select
import stat
from sys import stdin
import time
import webbrowser
from base64 import (
from six.moves.urllib.parse import parse_qs
from lazr.restfulclient.errors import HTTPError
from lazr.restfulclient.authorize.oauth import (
from launchpadlib import uris
class RequestTokenAuthorizationEngine(object):
    """The superclass of all request token authorizers.

    This base class does not implement request token authorization,
    since that varies depending on how you want the end-user to
    authorize a request token. You'll need to subclass this class and
    implement `make_end_user_authorize_token`.
    """
    UNAUTHORIZED_ACCESS_LEVEL = 'UNAUTHORIZED'

    def __init__(self, service_root, application_name=None, consumer_name=None, allow_access_levels=None):
        """Base class initialization.

        :param service_root: The root of the Launchpad instance being
            used.

        :param application_name: The name of the application that
            wants to use launchpadlib. This is used in conjunction
            with a desktop-wide integration.

            If you specify this argument, your values for
            consumer_name and allow_access_levels are ignored.

        :param consumer_name: The OAuth consumer name, for an
            application that wants its own point of integration into
            Launchpad. In almost all cases, you want to specify
            application_name instead and do a desktop-wide
            integration. The exception is when you're integrating a
            third-party website into Launchpad.

        :param allow_access_levels: A list of the Launchpad access
            levels to present to the user. ('READ_PUBLIC' and so on.)
            Your value for this argument will be ignored during a
            desktop-wide integration.
        :type allow_access_levels: A list of strings.
        """
        self.service_root = uris.lookup_service_root(service_root)
        self.web_root = uris.web_root_for_service_root(service_root)
        if application_name is None and consumer_name is None:
            raise ValueError('You must provide either application_name or consumer_name.')
        if application_name is not None and consumer_name is not None:
            raise ValueError('You must provide only one of application_name and consumer_name. (You provided %r and %r.)' % (application_name, consumer_name))
        if consumer_name is None:
            allow_access_levels = ['DESKTOP_INTEGRATION']
            consumer = SystemWideConsumer(application_name)
        else:
            consumer = Consumer(consumer_name)
            application_name = consumer_name
        self.consumer = consumer
        self.application_name = application_name
        self.allow_access_levels = allow_access_levels or []

    @property
    def unique_consumer_id(self):
        """Return a string identifying this consumer on this host."""
        return self.consumer.key + '@' + self.service_root

    def authorization_url(self, request_token):
        """Return the authorization URL for a request token.

        This is the URL the end-user must visit to authorize the
        token. How exactly does this happen? That depends on the
        subclass implementation.
        """
        page = '%s?oauth_token=%s' % (authorize_token_page, request_token)
        allow_permission = '&allow_permission='
        if len(self.allow_access_levels) > 0:
            page += allow_permission + allow_permission.join(self.allow_access_levels)
        return urljoin(self.web_root, page)

    def __call__(self, credentials, credential_store):
        """Authorize a token and associate it with the given credentials.

        If the credential store runs into a problem storing the
        credential locally, the `credential_save_failed` callback will
        be invoked. The callback will not be invoked if there's a
        problem authorizing the credentials.

        :param credentials: A `Credentials` object. If the end-user
            authorizes these credentials, this object will have its
            .access_token property set.

        :param credential_store: A `CredentialStore` object. If the
            end-user authorizes the credentials, they will be
            persisted locally using this object.

        :return: If the credentials are successfully authorized, the
            return value is the `Credentials` object originally passed
            in. Otherwise the return value is None.
        """
        request_token_string = self.get_request_token(credentials)
        self.make_end_user_authorize_token(credentials, request_token_string)
        if credentials.access_token is None:
            return None
        credential_store.save(credentials, self.unique_consumer_id)
        return credentials

    def get_request_token(self, credentials):
        """Get a new request token from the server.

        :param return: The request token.
        """
        authorization_json = credentials.get_request_token(web_root=self.web_root, token_format=Credentials.DICT_TOKEN_FORMAT)
        return authorization_json['oauth_token']

    def make_end_user_authorize_token(self, credentials, request_token):
        """Authorize the given request token using the given credentials.

        Your subclass must implement this method: it has no default
        implementation.

        Because an access token may expire or be revoked in the middle
        of a session, this method may be called at arbitrary points in
        a launchpadlib session, or even multiple times during a single
        session (with a different request token each time).

        In most cases, however, this method will be called at the
        beginning of a launchpadlib session, or not at all.
        """
        raise NotImplementedError()