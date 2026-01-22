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
def _credentials_need_refresh(self):
    if self.anon:
        return False
    if self._credential_expiry_time is None:
        return False
    else:
        delta = self._credential_expiry_time - datetime.utcnow()
        seconds_left = (delta.microseconds + (delta.seconds + delta.days * 24 * 3600) * 10 ** 6) / 10 ** 6
        if seconds_left < 5 * 60:
            boto.log.debug('Credentials need to be refreshed.')
            return True
        else:
            return False