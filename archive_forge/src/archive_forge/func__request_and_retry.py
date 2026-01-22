import atexit
import errno
import os
import re
import shutil
import sys
import tempfile
from hashlib import md5
from io import BytesIO
from json import dumps
from time import sleep
from httplib2 import Http, urlnorm
from wadllib.application import Application
from lazr.restfulclient._json import DatetimeJSONEncoder
from lazr.restfulclient.errors import HTTPError, error_for
from lazr.uri import URI
def _request_and_retry(self, url, method, body, headers):
    for retry_count in range(0, self.max_retries + 1):
        response, content = self._connection.request(url, method=method, body=body, headers=headers)
        if response.status in [502, 503] and retry_count < self.max_retries:
            sleep_for = int(2 ** (retry_count - 1))
            sleep(sleep_for)
        else:
            break
    return (response, content)