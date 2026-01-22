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
def _retrieve_credentials_using(self, credential_process):
    process_list = compat_shell_split(credential_process)
    p = self._popen(process_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        raise CredentialRetrievalError(provider=self.METHOD, error_msg=stderr.decode('utf-8'))
    parsed = botocore.compat.json.loads(stdout.decode('utf-8'))
    version = parsed.get('Version', '<Version key not provided>')
    if version != 1:
        raise CredentialRetrievalError(provider=self.METHOD, error_msg=f"Unsupported version '{version}' for credential process provider, supported versions: 1")
    try:
        return {'access_key': parsed['AccessKeyId'], 'secret_key': parsed['SecretAccessKey'], 'token': parsed.get('SessionToken'), 'expiry_time': parsed.get('Expiration')}
    except KeyError as e:
        raise CredentialRetrievalError(provider=self.METHOD, error_msg=f'Missing required key in response: {e}')