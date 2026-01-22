from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
class S3ObjectResource(resource_reference.ObjectResource):
    """API-specific subclass for handling metadata."""

    def __init__(self, storage_url_object, acl=None, cache_control=None, component_count=None, content_disposition=None, content_encoding=None, content_language=None, content_type=None, crc32c_hash=resource_reference.NOT_SUPPORTED_DO_NOT_DISPLAY, creation_time=None, custom_fields=None, custom_time=None, decryption_key_hash_sha256=None, encryption_algorithm=None, etag=None, event_based_hold=None, kms_key=None, md5_hash=None, metadata=None, metageneration=None, noncurrent_time=None, retention_expiration=None, size=None, storage_class=None, temporary_hold=None, update_time=None):
        """Initializes S3ObjectResource."""
        super(S3ObjectResource, self).__init__(storage_url_object, acl, cache_control, component_count, content_disposition, content_encoding, content_language, content_type, crc32c_hash, creation_time, custom_fields, custom_time, decryption_key_hash_sha256, encryption_algorithm, etag, event_based_hold, kms_key, md5_hash, metadata, metageneration, noncurrent_time, retention_expiration, size, storage_class, temporary_hold, update_time)

    def get_json_dump(self):
        return _get_json_dump(self)