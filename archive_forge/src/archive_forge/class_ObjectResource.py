from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_util
class ObjectResource(CloudResource):
    """Class representing a cloud object confirmed to exist.

  Warning: After being run through through output formatter utils (e.g. in
  `shim_format_util.py`), these fields may all be strings.

  Attributes:
    TYPE_STRING (str): String representing the resource's type.
    storage_url (StorageUrl): A StorageUrl object representing the object.
    scheme (storage_url.ProviderPrefix): Prefix indicating what cloud provider
      hosts the object.
    bucket (str): Bucket that contains the object.
    name (str): Name of object.
    generation (str|None): Generation (or "version") of the underlying object.
    acl (dict|str|None): ACLs dict or predefined-ACL string for the objects. If
      the API call to fetch the data failed, this can be an error string.
    cache_control (str|None): Describes the object's cache settings.
    component_count (int|None): Number of components, if any.
    content_disposition (str|None): Whether the object should be displayed or
      downloaded.
    content_encoding (str|None): Encodings that have been applied to the object.
    content_language (str|None): Language used in the object's content.
    content_type (str|None): A MIME type describing the object's content.
    custom_time (str|None): A timestamp in RFC 3339 format specified by the user
      for an object. Currently, GCS-only, but not in provider-specific class
      because generic daisy chain logic uses the field.
    crc32c_hash (str|None): Base64-encoded digest of crc32c hash.
    creation_time (datetime|None): Time the object was created.
    custom_fields (dict|None): Custom key-value pairs set by users.
    decryption_key_hash_sha256 (str|None): Digest of a customer-supplied
      encryption key.
    encryption_algorithm (str|None): Encryption algorithm used for encrypting
      the object if CSEK is used.
    etag (str|None): HTTP version identifier.
    event_based_hold (bool|None): Event based hold information for the object.
      Currently, GCS-only, but left generic because can affect copy logic.
    kms_key (str|None): Resource identifier of a Google-managed encryption key.
    md5_hash (str|None): Base64-encoded digest of md5 hash.
    metadata (object|dict|None): Cloud-specific metadata type.
    metageneration (int|None): Generation object's metadata.
    noncurrent_time (datetime|None): Noncurrent time value for the object.
    retention_expiration (datetime|None): Retention expiration information.
    size (int|None): Size of object in bytes (equivalent to content_length).
    storage_class (str|None): Storage class of the bucket.
    temporary_hold (bool|None): Temporary hold information for the object.
    update_time (datetime|None): Time the object was updated.
  """
    TYPE_STRING = 'cloud_object'

    def __init__(self, storage_url_object, acl=None, cache_control=None, component_count=None, content_disposition=None, content_encoding=None, content_language=None, content_type=None, crc32c_hash=None, creation_time=None, custom_fields=None, custom_time=None, decryption_key_hash_sha256=None, encryption_algorithm=None, etag=None, event_based_hold=None, kms_key=None, md5_hash=None, metadata=None, metageneration=None, noncurrent_time=None, retention_expiration=None, size=None, storage_class=None, temporary_hold=None, update_time=None):
        """Initializes resource. Args are a subset of attributes."""
        super(ObjectResource, self).__init__(storage_url_object)
        self.acl = acl
        self.cache_control = cache_control
        self.component_count = component_count
        self.content_disposition = content_disposition
        self.content_encoding = content_encoding
        self.content_language = content_language
        self.content_type = content_type
        self.crc32c_hash = crc32c_hash
        self.creation_time = creation_time
        self.custom_fields = custom_fields
        self.custom_time = custom_time
        self.decryption_key_hash_sha256 = decryption_key_hash_sha256
        self.encryption_algorithm = encryption_algorithm
        self.etag = etag
        self.event_based_hold = event_based_hold
        self.kms_key = kms_key
        self.md5_hash = md5_hash
        self.metageneration = metageneration
        self.metadata = metadata
        self.noncurrent_time = noncurrent_time
        self.retention_expiration = retention_expiration
        self.size = size
        self.storage_class = storage_class
        self.temporary_hold = temporary_hold
        self.update_time = update_time

    @property
    def bucket(self):
        return self.storage_url.bucket_name

    @property
    def name(self):
        return self.storage_url.object_name

    @property
    def generation(self):
        return self.storage_url.generation

    @property
    def is_symlink(self):
        """Returns whether this object is a symlink."""
        if not self.custom_fields or resource_util.SYMLINK_METADATA_KEY not in self.custom_fields:
            return False
        return self.custom_fields[resource_util.SYMLINK_METADATA_KEY].lower() == 'true'

    def __eq__(self, other):
        return super(ObjectResource, self).__eq__(other) and self.acl == other.acl and (self.cache_control == other.cache_control) and (self.component_count == other.component_count) and (self.content_disposition == other.content_disposition) and (self.content_encoding == other.content_encoding) and (self.content_language == other.content_language) and (self.content_type == other.content_type) and (self.crc32c_hash == other.crc32c_hash) and (self.creation_time == other.creation_time) and (self.custom_fields == other.custom_fields) and (self.custom_time == other.custom_time) and (self.decryption_key_hash_sha256 == other.decryption_key_hash_sha256) and (self.encryption_algorithm == other.encryption_algorithm) and (self.etag == other.etag) and (self.event_based_hold == other.event_based_hold) and (self.kms_key == other.kms_key) and (self.md5_hash == other.md5_hash) and (self.metadata == other.metadata) and (self.metageneration == other.metageneration) and (self.noncurrent_time == other.noncurrent_time) and (self.retention_expiration == other.retention_expiration) and (self.size == other.size) and (self.storage_class == other.storage_class) and (self.temporary_hold == other.temporary_hold) and (self.update_time == other.update_time)

    def is_container(self):
        return False

    def is_encrypted(self):
        raise NotImplementedError