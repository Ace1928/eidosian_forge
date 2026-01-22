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
class ProfileProviderBuilder:
    """This class handles the creation of profile based providers.

    NOTE: This class is only intended for internal use.

    This class handles the creation and ordering of the various credential
    providers that primarly source their configuration from the shared config.
    This is needed to enable sharing between the default credential chain and
    the source profile chain created by the assume role provider.
    """

    def __init__(self, session, cache=None, region_name=None, sso_token_cache=None):
        self._session = session
        self._cache = cache
        self._region_name = region_name
        self._sso_token_cache = sso_token_cache

    def providers(self, profile_name, disable_env_vars=False):
        return [self._create_web_identity_provider(profile_name, disable_env_vars), self._create_sso_provider(profile_name), self._create_shared_credential_provider(profile_name), self._create_process_provider(profile_name), self._create_config_provider(profile_name)]

    def _create_process_provider(self, profile_name):
        return ProcessProvider(profile_name=profile_name, load_config=lambda: self._session.full_config)

    def _create_shared_credential_provider(self, profile_name):
        credential_file = self._session.get_config_variable('credentials_file')
        return SharedCredentialProvider(profile_name=profile_name, creds_filename=credential_file)

    def _create_config_provider(self, profile_name):
        config_file = self._session.get_config_variable('config_file')
        return ConfigProvider(profile_name=profile_name, config_filename=config_file)

    def _create_web_identity_provider(self, profile_name, disable_env_vars):
        return AssumeRoleWithWebIdentityProvider(load_config=lambda: self._session.full_config, client_creator=_get_client_creator(self._session, self._region_name), cache=self._cache, profile_name=profile_name, disable_env_vars=disable_env_vars)

    def _create_sso_provider(self, profile_name):
        return SSOProvider(load_config=lambda: self._session.full_config, client_creator=self._session.create_client, profile_name=profile_name, cache=self._cache, token_cache=self._sso_token_cache, token_provider=SSOTokenProvider(self._session, cache=self._sso_token_cache, profile_name=profile_name))