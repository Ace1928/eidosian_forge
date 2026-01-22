import os
from boto.compat import six
from datetime import datetime
import boto
from boto import config
from boto.compat import expanduser
from boto.pyami.config import Config
from boto.exception import InvalidInstanceMetadataError
from boto.gs.acl import ACL
from boto.gs.acl import CannedACLStrings as CannedGSACLStrings
from boto.s3.acl import CannedACLStrings as CannedS3ACLStrings
from boto.s3.acl import Policy
def _populate_keys_from_metadata_server(self):
    boto.log.debug('Retrieving credentials from metadata server.')
    from boto.utils import get_instance_metadata
    timeout = config.getfloat('Boto', 'metadata_service_timeout', 1.0)
    attempts = config.getint('Boto', 'metadata_service_num_attempts', 1)
    metadata = get_instance_metadata(timeout=timeout, num_retries=attempts, data='meta-data/iam/security-credentials/')
    if metadata:
        creds = self._get_credentials_from_metadata(metadata)
        self._access_key = creds[0]
        self._secret_key = creds[1]
        self._security_token = creds[2]
        expires_at = creds[3]
        self._credential_expiry_time = datetime.strptime(expires_at, '%Y-%m-%dT%H:%M:%SZ')
        boto.log.debug('Retrieved credentials will expire in %s at: %s', self._credential_expiry_time - datetime.now(), expires_at)