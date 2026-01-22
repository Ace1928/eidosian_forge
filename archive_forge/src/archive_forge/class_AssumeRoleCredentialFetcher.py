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
class AssumeRoleCredentialFetcher(BaseAssumeRoleCredentialFetcher):

    def __init__(self, client_creator, source_credentials, role_arn, extra_args=None, mfa_prompter=None, cache=None, expiry_window_seconds=None):
        """
        :type client_creator: callable
        :param client_creator: A callable that creates a client taking
            arguments like ``Session.create_client``.

        :type source_credentials: Credentials
        :param source_credentials: The credentials to use to create the
            client for the call to AssumeRole.

        :type role_arn: str
        :param role_arn: The ARN of the role to be assumed.

        :type extra_args: dict
        :param extra_args: Any additional arguments to add to the assume
            role request using the format of the botocore operation.
            Possible keys include, but may not be limited to,
            DurationSeconds, Policy, SerialNumber, ExternalId and
            RoleSessionName.

        :type mfa_prompter: callable
        :param mfa_prompter: A callable that returns input provided by the
            user (i.e raw_input, getpass.getpass, etc.).

        :type cache: dict
        :param cache: An object that supports ``__getitem__``,
            ``__setitem__``, and ``__contains__``.  An example of this is
            the ``JSONFileCache`` class in aws-cli.

        :type expiry_window_seconds: int
        :param expiry_window_seconds: The amount of time, in seconds,
        """
        self._source_credentials = source_credentials
        self._mfa_prompter = mfa_prompter
        if self._mfa_prompter is None:
            self._mfa_prompter = getpass.getpass
        super().__init__(client_creator, role_arn, extra_args=extra_args, cache=cache, expiry_window_seconds=expiry_window_seconds)

    def _get_credentials(self):
        """Get credentials by calling assume role."""
        kwargs = self._assume_role_kwargs()
        client = self._create_client()
        return client.assume_role(**kwargs)

    def _assume_role_kwargs(self):
        """Get the arguments for assume role based on current configuration."""
        assume_role_kwargs = deepcopy(self._assume_kwargs)
        mfa_serial = assume_role_kwargs.get('SerialNumber')
        if mfa_serial is not None:
            prompt = 'Enter MFA code for %s: ' % mfa_serial
            token_code = self._mfa_prompter(prompt)
            assume_role_kwargs['TokenCode'] = token_code
        duration_seconds = assume_role_kwargs.get('DurationSeconds')
        if duration_seconds is not None:
            assume_role_kwargs['DurationSeconds'] = duration_seconds
        return assume_role_kwargs

    def _create_client(self):
        """Create an STS client using the source credentials."""
        frozen_credentials = self._source_credentials.get_frozen_credentials()
        return self._client_creator('sts', aws_access_key_id=frozen_credentials.access_key, aws_secret_access_key=frozen_credentials.secret_key, aws_session_token=frozen_credentials.token)