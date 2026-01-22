import datetime
import getpass
import json
import logging
import os
import subprocess
import threading
import time
from collections import namedtuple
from copy import deepcopy
from hashlib import sha1
from dateutil.parser import parse
from dateutil.tz import tzlocal, tzutc
import botocore.compat
import botocore.configloader
from botocore import UNSIGNED
from botocore.compat import compat_shell_split, total_seconds
from botocore.config import Config
from botocore.exceptions import (
from botocore.tokens import SSOTokenProvider
from botocore.utils import (
class RefreshableCredentials(Credentials):
    """
    Holds the credentials needed to authenticate requests. In addition, it
    knows how to refresh itself.

    :param str access_key: The access key part of the credentials.
    :param str secret_key: The secret key part of the credentials.
    :param str token: The security token, valid only for session credentials.
    :param datetime expiry_time: The expiration time of the credentials.
    :param function refresh_using: Callback function to refresh the credentials.
    :param str method: A string which identifies where the credentials
        were found.
    :param function time_fetcher: Callback function to retrieve current time.
    """
    _advisory_refresh_timeout = _DEFAULT_ADVISORY_REFRESH_TIMEOUT
    _mandatory_refresh_timeout = _DEFAULT_MANDATORY_REFRESH_TIMEOUT

    def __init__(self, access_key, secret_key, token, expiry_time, refresh_using, method, time_fetcher=_local_now, advisory_timeout=None, mandatory_timeout=None):
        self._refresh_using = refresh_using
        self._access_key = access_key
        self._secret_key = secret_key
        self._token = token
        self._expiry_time = expiry_time
        self._time_fetcher = time_fetcher
        self._refresh_lock = threading.Lock()
        self.method = method
        self._frozen_credentials = ReadOnlyCredentials(access_key, secret_key, token)
        self._normalize()
        if advisory_timeout is not None:
            self._advisory_refresh_timeout = advisory_timeout
        if mandatory_timeout is not None:
            self._mandatory_refresh_timeout = mandatory_timeout

    def _normalize(self):
        self._access_key = botocore.compat.ensure_unicode(self._access_key)
        self._secret_key = botocore.compat.ensure_unicode(self._secret_key)

    @classmethod
    def create_from_metadata(cls, metadata, refresh_using, method, advisory_timeout=None, mandatory_timeout=None):
        kwargs = {}
        if advisory_timeout is not None:
            kwargs['advisory_timeout'] = advisory_timeout
        if mandatory_timeout is not None:
            kwargs['mandatory_timeout'] = mandatory_timeout
        instance = cls(access_key=metadata['access_key'], secret_key=metadata['secret_key'], token=metadata['token'], expiry_time=cls._expiry_datetime(metadata['expiry_time']), method=method, refresh_using=refresh_using, **kwargs)
        return instance

    @property
    def access_key(self):
        """Warning: Using this property can lead to race conditions if you
        access another property subsequently along the refresh boundary.
        Please use get_frozen_credentials instead.
        """
        self._refresh()
        return self._access_key

    @access_key.setter
    def access_key(self, value):
        self._access_key = value

    @property
    def secret_key(self):
        """Warning: Using this property can lead to race conditions if you
        access another property subsequently along the refresh boundary.
        Please use get_frozen_credentials instead.
        """
        self._refresh()
        return self._secret_key

    @secret_key.setter
    def secret_key(self, value):
        self._secret_key = value

    @property
    def token(self):
        """Warning: Using this property can lead to race conditions if you
        access another property subsequently along the refresh boundary.
        Please use get_frozen_credentials instead.
        """
        self._refresh()
        return self._token

    @token.setter
    def token(self, value):
        self._token = value

    def _seconds_remaining(self):
        delta = self._expiry_time - self._time_fetcher()
        return total_seconds(delta)

    def refresh_needed(self, refresh_in=None):
        """Check if a refresh is needed.

        A refresh is needed if the expiry time associated
        with the temporary credentials is less than the
        provided ``refresh_in``.  If ``time_delta`` is not
        provided, ``self.advisory_refresh_needed`` will be used.

        For example, if your temporary credentials expire
        in 10 minutes and the provided ``refresh_in`` is
        ``15 * 60``, then this function will return ``True``.

        :type refresh_in: int
        :param refresh_in: The number of seconds before the
            credentials expire in which refresh attempts should
            be made.

        :return: True if refresh needed, False otherwise.

        """
        if self._expiry_time is None:
            return False
        if refresh_in is None:
            refresh_in = self._advisory_refresh_timeout
        if self._seconds_remaining() >= refresh_in:
            return False
        logger.debug('Credentials need to be refreshed.')
        return True

    def _is_expired(self):
        return self.refresh_needed(refresh_in=0)

    def _refresh(self):
        if not self.refresh_needed(self._advisory_refresh_timeout):
            return
        if self._refresh_lock.acquire(False):
            try:
                if not self.refresh_needed(self._advisory_refresh_timeout):
                    return
                is_mandatory_refresh = self.refresh_needed(self._mandatory_refresh_timeout)
                self._protected_refresh(is_mandatory=is_mandatory_refresh)
                return
            finally:
                self._refresh_lock.release()
        elif self.refresh_needed(self._mandatory_refresh_timeout):
            with self._refresh_lock:
                if not self.refresh_needed(self._mandatory_refresh_timeout):
                    return
                self._protected_refresh(is_mandatory=True)

    def _protected_refresh(self, is_mandatory):
        try:
            metadata = self._refresh_using()
        except Exception:
            period_name = 'mandatory' if is_mandatory else 'advisory'
            logger.warning('Refreshing temporary credentials failed during %s refresh period.', period_name, exc_info=True)
            if is_mandatory:
                raise
            return
        self._set_from_data(metadata)
        self._frozen_credentials = ReadOnlyCredentials(self._access_key, self._secret_key, self._token)
        if self._is_expired():
            msg = 'Credentials were refreshed, but the refreshed credentials are still expired.'
            logger.warning(msg)
            raise RuntimeError(msg)

    @staticmethod
    def _expiry_datetime(time_str):
        return parse(time_str)

    def _set_from_data(self, data):
        expected_keys = ['access_key', 'secret_key', 'token', 'expiry_time']
        if not data:
            missing_keys = expected_keys
        else:
            missing_keys = [k for k in expected_keys if k not in data]
        if missing_keys:
            message = 'Credential refresh failed, response did not contain: %s'
            raise CredentialRetrievalError(provider=self.method, error_msg=message % ', '.join(missing_keys))
        self.access_key = data['access_key']
        self.secret_key = data['secret_key']
        self.token = data['token']
        self._expiry_time = parse(data['expiry_time'])
        logger.debug('Retrieved credentials will expire at: %s', self._expiry_time)
        self._normalize()

    def get_frozen_credentials(self):
        """Return immutable credentials.

        The ``access_key``, ``secret_key``, and ``token`` properties
        on this class will always check and refresh credentials if
        needed before returning the particular credentials.

        This has an edge case where you can get inconsistent
        credentials.  Imagine this:

            # Current creds are "t1"
            tmp.access_key  ---> expired? no, so return t1.access_key
            # ---- time is now expired, creds need refreshing to "t2" ----
            tmp.secret_key  ---> expired? yes, refresh and return t2.secret_key

        This means we're using the access key from t1 with the secret key
        from t2.  To fix this issue, you can request a frozen credential object
        which is guaranteed not to change.

        The frozen credentials returned from this method should be used
        immediately and then discarded.  The typical usage pattern would
        be::

            creds = RefreshableCredentials(...)
            some_code = SomeSignerObject()
            # I'm about to sign the request.
            # The frozen credentials are only used for the
            # duration of generate_presigned_url and will be
            # immediately thrown away.
            request = some_code.sign_some_request(
                with_credentials=creds.get_frozen_credentials())
            print("Signed request:", request)

        """
        self._refresh()
        return self._frozen_credentials