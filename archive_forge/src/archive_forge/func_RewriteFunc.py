from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
import time
from apitools.base.py import encoding
from boto import config
from gslib.cloud_api import EncryptionException
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.name_expansion import NameExpansionIterator
from gslib.name_expansion import SeekAheadNameExpansionIterator
from gslib.progress_callback import FileProgressCallbackHandler
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.thread_message import FileMessage
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.constants import NO_MAX
from gslib.utils.constants import UTF8
from gslib.utils.encryption_helper import CryptoKeyType
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
from gslib.utils.encryption_helper import GetEncryptionKeyWrapper
from gslib.utils.encryption_helper import MAX_DECRYPTION_KEYS
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.system_util import StdinIterator
from gslib.utils.text_util import ConvertRecursiveToFlatWildcard
from gslib.utils.text_util import NormalizeStorageClass
from gslib.utils import text_util
from gslib.utils.translation_helper import PreconditionsFromHeaders
def RewriteFunc(self, name_expansion_result, thread_state=None):
    gsutil_api = GetCloudApiInstance(self, thread_state=thread_state)
    transform_url = name_expansion_result.expanded_storage_url
    self.CheckProvider(transform_url)
    src_metadata = gsutil_api.GetObjectMetadata(transform_url.bucket_name, transform_url.object_name, generation=transform_url.generation, provider=transform_url.scheme)
    if self.no_preserve_acl:
        src_metadata.acl = []
    elif not src_metadata.acl:
        raise CommandException("No OWNER permission found for object %s. If your bucket has uniform bucket-level access (UBLA) enabled, include the -O option in your command to avoid this error. If your bucket does not use UBLA, you can use the -O option to apply the bucket's default object ACL when rewriting." % transform_url)
    src_encryption_kms_key = src_metadata.kmsKeyName if src_metadata.kmsKeyName else None
    src_encryption_sha256 = None
    if src_metadata.customerEncryption and src_metadata.customerEncryption.keySha256:
        src_encryption_sha256 = src_metadata.customerEncryption.keySha256
        src_encryption_sha256 = src_encryption_sha256.encode('ascii')
    src_was_encrypted = src_encryption_sha256 is not None or src_encryption_kms_key is not None
    dest_encryption_kms_key = None
    if self.boto_file_encryption_keywrapper is not None and self.boto_file_encryption_keywrapper.crypto_type == CryptoKeyType.CMEK:
        dest_encryption_kms_key = self.boto_file_encryption_keywrapper.crypto_key
    dest_encryption_sha256 = None
    if self.boto_file_encryption_keywrapper is not None and self.boto_file_encryption_keywrapper.crypto_type == CryptoKeyType.CSEK:
        dest_encryption_sha256 = self.boto_file_encryption_keywrapper.crypto_key_sha256
    should_encrypt_dest = self.boto_file_encryption_keywrapper is not None
    encryption_unchanged = src_encryption_sha256 == dest_encryption_sha256 and src_encryption_kms_key == dest_encryption_kms_key
    if _TransformTypes.CRYPTO_KEY not in self.transform_types and (not encryption_unchanged):
        raise EncryptionException('The "-k" flag was not passed to the rewrite command, but the encryption_key value in your boto config file did not match the key used to encrypt the object "%s" (hash: %s). To encrypt the object using a different key, you must specify the "-k" flag.' % (transform_url, src_encryption_sha256))
    redundant_transforms = []
    if _TransformTypes.STORAGE_CLASS in self.transform_types and self.dest_storage_class == NormalizeStorageClass(src_metadata.storageClass):
        redundant_transforms.append('storage class')
    if _TransformTypes.CRYPTO_KEY in self.transform_types and should_encrypt_dest and encryption_unchanged:
        redundant_transforms.append('encryption key')
    if len(redundant_transforms) == len(self.transform_types):
        self.logger.info('Skipping %s, all transformations were redundant: %s' % (transform_url, redundant_transforms))
        return
    dest_metadata = encoding.PyValueToMessage(apitools_messages.Object, encoding.MessageToPyValue(src_metadata))
    dest_metadata.generation = None
    dest_metadata.id = None
    dest_metadata.customerEncryption = None
    dest_metadata.kmsKeyName = None
    if _TransformTypes.STORAGE_CLASS in self.transform_types:
        dest_metadata.storageClass = self.dest_storage_class
    if dest_encryption_kms_key is not None:
        dest_metadata.kmsKeyName = dest_encryption_kms_key
    decryption_keywrapper = None
    if src_encryption_sha256 is not None:
        if src_encryption_sha256 in self.csek_hash_to_keywrapper:
            decryption_keywrapper = self.csek_hash_to_keywrapper[src_encryption_sha256]
        else:
            raise EncryptionException('Missing decryption key with SHA256 hash %s. No decryption key matches object %s' % (src_encryption_sha256, transform_url))
    operation_name = 'Rewriting'
    if _TransformTypes.CRYPTO_KEY in self.transform_types:
        if src_was_encrypted and should_encrypt_dest:
            if not encryption_unchanged:
                operation_name = 'Rotating'
        elif src_was_encrypted and (not should_encrypt_dest):
            operation_name = 'Decrypting'
        elif not src_was_encrypted and should_encrypt_dest:
            operation_name = 'Encrypting'
    sys.stderr.write(_ConstructAnnounceText(operation_name, transform_url.url_string))
    sys.stderr.flush()
    gsutil_api.status_queue.put(FileMessage(transform_url, None, time.time(), finished=False, size=src_metadata.size, message_type=FileMessage.FILE_REWRITE))
    progress_callback = FileProgressCallbackHandler(gsutil_api.status_queue, src_url=transform_url, operation_name=operation_name).call
    gsutil_api.CopyObject(src_metadata, dest_metadata, src_generation=transform_url.generation, preconditions=self.preconditions, progress_callback=progress_callback, decryption_tuple=decryption_keywrapper, encryption_tuple=self.boto_file_encryption_keywrapper, provider=transform_url.scheme, fields=[])
    gsutil_api.status_queue.put(FileMessage(transform_url, None, time.time(), finished=True, size=src_metadata.size, message_type=FileMessage.FILE_REWRITE))