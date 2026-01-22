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
class InstanceMetadataProvider(CredentialProvider):
    METHOD = 'iam-role'
    CANONICAL_NAME = 'Ec2InstanceMetadata'

    def __init__(self, iam_role_fetcher):
        self._role_fetcher = iam_role_fetcher

    def load(self):
        fetcher = self._role_fetcher
        metadata = fetcher.retrieve_iam_role_credentials()
        if not metadata:
            return None
        logger.info('Found credentials from IAM Role: %s', metadata['role_name'])
        creds = RefreshableCredentials.create_from_metadata(metadata, method=self.METHOD, refresh_using=fetcher.retrieve_iam_role_credentials)
        return creds