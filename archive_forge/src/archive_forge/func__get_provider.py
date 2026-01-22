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
def _get_provider(self, canonical_name):
    """Return a credential provider by its canonical name.

        :type canonical_name: str
        :param canonical_name: The canonical name of the provider.

        :raises UnknownCredentialError: Raised if no
            credential provider by the provided name
            is found.
        """
    provider = self._get_provider_by_canonical_name(canonical_name)
    if canonical_name.lower() in ['sharedconfig', 'sharedcredentials']:
        assume_role_provider = self._get_provider_by_method('assume-role')
        if assume_role_provider is not None:
            if provider is None:
                return assume_role_provider
            return CredentialResolver([assume_role_provider, provider])
    if provider is None:
        raise UnknownCredentialError(name=canonical_name)
    return provider