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
class AuthorizeRequestTokenWithBrowser(AuthorizeRequestTokenWithURL):
    """Authorize using a URL that pops-up automatically in a browser.

    This authorizer simply opens up the end-user's web browser to a
    Launchpad URL and lets the end-user authorize the request token
    themselves.

    This is the same as its superclass, except this class also
    performs the browser automatic opening of the URL.
    """
    WAITING_FOR_USER = 'The authorization page:\n (%s)\nshould be opening in your browser. Use your browser to authorize\nthis program to access Launchpad on your behalf.'
    TIMEOUT_MESSAGE = 'Press Enter to continue or wait (%d) seconds...'
    TIMEOUT = 5
    TERMINAL_BROWSERS = ('www-browser', 'links', 'links2', 'lynx', 'elinks', 'elinks-lite', 'netrik', 'w3m')
    WAITING_FOR_LAUNCHPAD = 'Waiting to hear from Launchpad about your decision...'

    def __init__(self, service_root, application_name, consumer_name=None, credential_save_failed=None, allow_access_levels=None):
        """Constructor.

        :param service_root: See `RequestTokenAuthorizationEngine`.
        :param application_name: See `RequestTokenAuthorizationEngine`.
        :param consumer_name: The value of this argument is
            ignored. If we have the capability to open the end-user's
            web browser, we must be running on the end-user's computer,
            so we should do a full desktop integration.
        :param credential_save_failed: See `RequestTokenAuthorizationEngine`.
        :param allow_access_levels: The value of this argument is
            ignored, for the same reason as consumer_name.
        """
        super(AuthorizeRequestTokenWithBrowser, self).__init__(service_root, application_name, None, credential_save_failed)

    def notify_end_user_authorization_url(self, authorization_url):
        """Notify the end-user of the URL."""
        super(AuthorizeRequestTokenWithBrowser, self).notify_end_user_authorization_url(authorization_url)
        try:
            browser_obj = webbrowser.get()
            browser = getattr(browser_obj, 'basename', None)
            console_browser = browser in self.TERMINAL_BROWSERS
        except webbrowser.Error:
            browser_obj = None
            console_browser = False
        if console_browser:
            self.output(self.TIMEOUT_MESSAGE % self.TIMEOUT)
            rlist, _, _ = select([stdin], [], [], self.TIMEOUT)
            if rlist:
                stdin.readline()
        if browser_obj is not None:
            webbrowser.open(authorization_url)

    def wait_for_end_user_authorization(self, credentials):
        """Wait for the end-user to authorize"""
        self.output(self.WAITING_FOR_LAUNCHPAD)
        start_time = time.time()
        while credentials.access_token is None:
            time.sleep(access_token_poll_time)
            try:
                if self.check_end_user_authorization(credentials):
                    break
            except EndUserNoAuthorization:
                pass
            if time.time() >= start_time + access_token_poll_timeout:
                raise TokenAuthorizationTimedOut('Timed out after %d seconds.' % access_token_poll_timeout)