import uuid
import base64
import time
from boto.compat import six, json
from boto.cloudfront.identity import OriginAccessIdentity
from boto.cloudfront.object import Object, StreamingObject
from boto.cloudfront.signers import ActiveTrustedSigners, TrustedSigners
from boto.cloudfront.logging import LoggingInfo
from boto.cloudfront.origin import S3Origin, CustomOrigin
from boto.s3.acl import ACL
@staticmethod
def _canned_policy(resource, expires):
    """
        Creates a canned policy string.
        """
    policy = '{"Statement":[{"Resource":"%(resource)s","Condition":{"DateLessThan":{"AWS:EpochTime":%(expires)s}}}]}' % locals()
    return policy