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
def _load_authentication(self):
    """Read authentication from kube-config user section if exists.

        This function goes through various authentication methods in user
        section of kube-config and stops if it finds a valid authentication
        method. The order of authentication methods is:

            1. auth-provider (gcp, azure, oidc)
            2. token field (point to a token file)
            3. exec provided plugin
            4. username/password
        """
    if not self._user:
        return
    if self._load_auth_provider_token():
        return
    if self._load_user_token():
        return
    if self._load_from_exec_plugin():
        return
    self._load_user_pass_token()