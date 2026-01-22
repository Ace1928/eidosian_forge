import atexit
import base64
import copy
import datetime
import json
import logging
import os
import platform
import tempfile
import time
import google.auth
import google.auth.transport.requests
import oauthlib.oauth2
import urllib3
from ruamel import yaml
from requests_oauthlib import OAuth2Session
from six import PY3
from kubernetes.client import ApiClient, Configuration
from kubernetes.config.exec_provider import ExecProvider
from .config_exception import ConfigException
from .dateutil import UTC, format_rfc3339, parse_rfc3339
class FileOrData(object):
    """Utility class to read content of obj[%data_key_name] or file's

     content of obj[%file_key_name] and represent it as file or data.
     Note that the data is preferred. The obj[%file_key_name] will be used iff
     obj['%data_key_name'] is not set or empty. Assumption is file content is
     raw data and data field is base64 string. The assumption can be changed
     with base64_file_content flag. If set to False, the content of the file
     will assumed to be base64 and read as is. The default True value will
     result in base64 encode of the file content after read.
  """

    def __init__(self, obj, file_key_name, data_key_name=None, file_base_path='', base64_file_content=True):
        if not data_key_name:
            data_key_name = file_key_name + '-data'
        self._file = None
        self._data = None
        self._base64_file_content = base64_file_content
        if data_key_name in obj:
            self._data = obj[data_key_name]
        elif file_key_name in obj:
            self._file = os.path.normpath(os.path.join(file_base_path, obj[file_key_name]))

    def as_file(self):
        """If obj[%data_key_name] exists, return name of a file with base64

        decoded obj[%data_key_name] content otherwise obj[%file_key_name].
    """
        use_data_if_no_file = not self._file and self._data
        if use_data_if_no_file:
            if self._base64_file_content:
                if isinstance(self._data, str):
                    content = self._data.encode()
                else:
                    content = self._data
                self._file = _create_temp_file_with_content(base64.standard_b64decode(content))
            else:
                self._file = _create_temp_file_with_content(self._data)
        if self._file and (not os.path.isfile(self._file)):
            raise ConfigException('File does not exists: %s' % self._file)
        return self._file

    def as_data(self):
        """If obj[%data_key_name] exists, Return obj[%data_key_name] otherwise

        base64 encoded string of obj[%file_key_name] file content.
    """
        use_file_if_no_data = not self._data and self._file
        if use_file_if_no_data:
            with open(self._file) as f:
                if self._base64_file_content:
                    self._data = bytes.decode(base64.standard_b64encode(str.encode(f.read())))
                else:
                    self._data = f.read()
        return self._data