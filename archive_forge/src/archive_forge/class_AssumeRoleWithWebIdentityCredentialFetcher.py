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
class AssumeRoleWithWebIdentityCredentialFetcher(BaseAssumeRoleCredentialFetcher):

    def __init__(self, client_creator, web_identity_token_loader, role_arn, extra_args=None, cache=None, expiry_window_seconds=None):
        """
        :type client_creator: callable
        :param client_creator: A callable that creates a client taking
            arguments like ``Session.create_client``.

        :type web_identity_token_loader: callable
        :param web_identity_token_loader: A callable that takes no arguments
        and returns a web identity token str.

        :type role_arn: str
        :param role_arn: The ARN of the role to be assumed.

        :type extra_args: dict
        :param extra_args: Any additional arguments to add to the assume
            role request using the format of the botocore operation.
            Possible keys include, but may not be limited to,
            DurationSeconds, Policy, SerialNumber, ExternalId and
            RoleSessionName.

        :type cache: dict
        :param cache: An object that supports ``__getitem__``,
            ``__setitem__``, and ``__contains__``.  An example of this is
            the ``JSONFileCache`` class in aws-cli.

        :type expiry_window_seconds: int
        :param expiry_window_seconds: The amount of time, in seconds,
        """
        self._web_identity_token_loader = web_identity_token_loader
        super().__init__(client_creator, role_arn, extra_args=extra_args, cache=cache, expiry_window_seconds=expiry_window_seconds)

    def _get_credentials(self):
        """Get credentials by calling assume role."""
        kwargs = self._assume_role_kwargs()
        config = Config(signature_version=UNSIGNED)
        client = self._client_creator('sts', config=config)
        return client.assume_role_with_web_identity(**kwargs)

    def _assume_role_kwargs(self):
        """Get the arguments for assume role based on current configuration."""
        assume_role_kwargs = deepcopy(self._assume_kwargs)
        identity_token = self._web_identity_token_loader()
        assume_role_kwargs['WebIdentityToken'] = identity_token
        return assume_role_kwargs