import os
import sys
import time
import errno
import base64
import logging
import datetime
import urllib.parse
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from libcloud.utils.py3 import b, httplib, urlparse, urlencode
from libcloud.common.base import BaseDriver, JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError
from libcloud.utils.connection import get_response_object
class GoogleServiceAcctAuthConnection(GoogleBaseAuthConnection):
    """Authentication class for "Service Account" authentication."""

    def __init__(self, user_id, key, *args, **kwargs):
        """
        Check to see if cryptography is available, and convert PEM key file
        into a key string, or extract the key from JSON object, string or
        file.

        :param  user_id: Email address to be used for Service Account
                authentication.
        :type   user_id: ``str``

        :param  key: The path to a PEM/JSON file containing the private RSA
        key, or a str/dict containing the PEM/JSON.
        :type   key: ``str`` or ``dict``

        """
        if SHA256 is None:
            raise GoogleAuthError('cryptography library required for Service Account Authentication.')
        if isinstance(key, dict):
            key_content = key
            key = None
        else:
            key_path = os.path.expanduser(key)
            if os.path.exists(key_path) and os.path.isfile(key_path):
                try:
                    with open(key_path) as f:
                        key_content = f.read()
                except OSError:
                    raise GoogleAuthError("Missing (or unreadable) key file: '%s'" % key)
            else:
                key_content = key
                key = None
        try:
            key_content = json.loads(key_content)
        except TypeError:
            pass
        except ValueError:
            pass
        finally:
            if 'private_key' in key_content:
                key = key_content['private_key']
            else:
                key = key_content
        try:
            serialization.load_pem_private_key(b(key), password=None, backend=default_backend())
        except ValueError as e:
            raise GoogleAuthError('Unable to decode provided PEM key: %s' % e)
        except TypeError as e:
            raise GoogleAuthError('Unable to decode provided PEM key: %s' % e)
        except exceptions.UnsupportedAlgorithm as e:
            raise GoogleAuthError('Unable to decode provided PEM key: %s' % e)
        super().__init__(user_id, key, *args, **kwargs)

    def get_new_token(self):
        """
        Get a new token using the email address and RSA Key.

        :return:  Dictionary containing token information
        :rtype:   ``dict``
        """
        header = {'alg': 'RS256', 'typ': 'JWT'}
        header_enc = base64.urlsafe_b64encode(b(json.dumps(header)))
        claim_set = {'iss': self.user_id, 'scope': self.scopes, 'aud': 'https://accounts.google.com/o/oauth2/token', 'exp': int(time.time()) + 3600, 'iat': int(time.time())}
        claim_set_enc = base64.urlsafe_b64encode(b(json.dumps(claim_set)))
        message = b'.'.join((header_enc, claim_set_enc))
        key = serialization.load_pem_private_key(b(self.key), password=None, backend=default_backend())
        signature = key.sign(data=b(message), padding=PKCS1v15(), algorithm=SHA256())
        signature = base64.urlsafe_b64encode(signature)
        jwt = b'.'.join((message, signature))
        request = {'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer', 'assertion': jwt}
        return self._token_request(request)