import json
import logging
import time
from oauth2client import _helpers
from oauth2client import _pure_python_crypt
def _bad_pkcs12_key_as_pem(*args, **kwargs):
    raise NotImplementedError('pkcs12_key_as_pem requires OpenSSL.')