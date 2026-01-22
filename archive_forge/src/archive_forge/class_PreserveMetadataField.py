from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import arg_parsers
class PreserveMetadataField(enum.Enum):
    ACL = 'acl'
    GID = 'gid'
    KMS_KEY = 'kms-key'
    MODE = 'mode'
    STORAGE_CLASS = 'storage-class'
    SYMLINK = 'symlink'
    TEMPORARY_HOLD = 'temporary-hold'
    TIME_CREATED = 'time-created'
    UID = 'uid'