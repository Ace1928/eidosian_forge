import json
import logging
import os
import threading
from datetime import datetime, timedelta
from typing import NamedTuple, Optional
import dateutil.parser
from dateutil.tz import tzutc
from botocore import UNSIGNED
from botocore.compat import total_seconds
from botocore.config import Config
from botocore.exceptions import (
from botocore.utils import CachedProperty, JSONFileCache, SSOTokenLoader
def _attempt_create_token(self, token):
    response = self._client.create_token(grantType=self._GRANT_TYPE, clientId=token['clientId'], clientSecret=token['clientSecret'], refreshToken=token['refreshToken'])
    expires_in = timedelta(seconds=response['expiresIn'])
    new_token = {'startUrl': self._sso_config['sso_start_url'], 'region': self._sso_config['sso_region'], 'accessToken': response['accessToken'], 'expiresAt': self._now() + expires_in, 'clientId': token['clientId'], 'clientSecret': token['clientSecret'], 'registrationExpiresAt': token['registrationExpiresAt']}
    if 'refreshToken' in response:
        new_token['refreshToken'] = response['refreshToken']
    logger.info('SSO Token refresh succeeded')
    return new_token