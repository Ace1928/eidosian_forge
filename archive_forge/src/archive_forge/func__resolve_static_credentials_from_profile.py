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
def _resolve_static_credentials_from_profile(self, profile):
    try:
        return Credentials(access_key=profile['aws_access_key_id'], secret_key=profile['aws_secret_access_key'], token=profile.get('aws_session_token'))
    except KeyError as e:
        raise PartialCredentialsError(provider=self.METHOD, cred_var=str(e))