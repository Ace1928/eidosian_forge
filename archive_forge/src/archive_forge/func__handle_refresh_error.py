import datetime
import json
import logging
import urllib
from oauth2client import _helpers
from oauth2client import client
from oauth2client import transport
from oauth2client.contrib import reauth
from oauth2client.contrib import reauth_errors
from six.moves import http_client
def _handle_refresh_error(self, http, rapt_refreshed, resp, content):
    d = {}
    try:
        d = json.loads(content)
    except (TypeError, ValueError):
        pass
    if not rapt_refreshed and d.get('error') == REAUTH_NEEDED_ERROR and (d.get('error_subtype') == REAUTH_NEEDED_ERROR_INVALID_RAPT or d.get('error_subtype') == REAUTH_NEEDED_ERROR_RAPT_REQUIRED):
        self.rapt_token = reauth.GetRaptToken(getattr(http, 'request', http), self.client_id, self.client_secret, self.refresh_token, self.token_uri, scopes=list(self.scopes))
        self._do_refresh_request(http, rapt_refreshed=True)
        return
    logger.info('Failed to retrieve access token: {0}'.format(content))
    error_msg = 'Invalid response {0}.'.format(resp.status)
    if 'error' in d:
        error_msg = d['error']
        if 'error_description' in d:
            error_msg += ': ' + d['error_description']
        self.invalid = True
        if self.store is not None:
            self.store.locked_put(self)
    raise reauth_errors.HttpAccessTokenRefreshError(error_msg, status=resp.status)