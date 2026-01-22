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
class RewriteCommand(Command):
    """Implementation of gsutil rewrite command."""
    command_spec = Command.CreateCommandSpec('rewrite', command_name_aliases=[], usage_synopsis=_SYNOPSIS, min_args=0, max_args=NO_MAX, supported_sub_args='fkIrROs:', file_url_ok=False, provider_url_ok=False, urls_start_arg=0, gs_api_support=[ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments=[CommandArgument.MakeZeroOrMoreCloudURLsArgument()])
    help_spec = Command.HelpSpec(help_name='rewrite', help_name_aliases=['rekey', 'rotate'], help_type='command_help', help_one_line_summary='Rewrite objects', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={})
    gcloud_storage_map = GcloudStorageMap(gcloud_command=['storage', 'objects', 'update'], flag_map={'-I': GcloudStorageFlag('-I'), '-f': GcloudStorageFlag('--continue-on-error'), '-k': None if config.get('GSUtil', 'encryption_key', None) else GcloudStorageFlag('--clear-encryption-key'), '-O': GcloudStorageFlag('--no-preserve-acl'), '-r': GcloudStorageFlag('-r'), '-R': GcloudStorageFlag('-r'), '-s': GcloudStorageFlag('-s')})

    def CheckProvider(self, url):
        if url.scheme != 'gs':
            raise CommandException('"rewrite" called on URL with unsupported provider: %s' % str(url))

    def RunCommand(self):
        """Command entry point for the rewrite command."""
        self.continue_on_error = self.parallel_operations
        self.csek_hash_to_keywrapper = {}
        self.dest_storage_class = None
        self.no_preserve_acl = False
        self.read_args_from_stdin = False
        self.supported_transformation_flags = ['-k', '-s']
        self.transform_types = set()
        self.op_failure_count = 0
        self.boto_file_encryption_keywrapper = GetEncryptionKeyWrapper(config)
        self.boto_file_encryption_sha256 = self.boto_file_encryption_keywrapper.crypto_key_sha256 if self.boto_file_encryption_keywrapper else None
        if self.sub_opts:
            for o, a in self.sub_opts:
                if o == '-f':
                    self.continue_on_error = True
                elif o == '-k':
                    self.transform_types.add(_TransformTypes.CRYPTO_KEY)
                elif o == '-I':
                    self.read_args_from_stdin = True
                elif o == '-O':
                    self.no_preserve_acl = True
                elif o == '-r' or o == '-R':
                    self.recursion_requested = True
                    self.all_versions = True
                elif o == '-s':
                    self.transform_types.add(_TransformTypes.STORAGE_CLASS)
                    self.dest_storage_class = NormalizeStorageClass(a)
        if self.read_args_from_stdin:
            if self.args:
                raise CommandException('No arguments allowed with the -I flag.')
            url_strs = StdinIterator()
        else:
            if not self.args:
                raise CommandException('The rewrite command (without -I) expects at least one URL.')
            url_strs = self.args
        if not self.transform_types:
            raise CommandException('rewrite command requires at least one transformation flag. Currently supported transformation flags: %s' % self.supported_transformation_flags)
        self.preconditions = PreconditionsFromHeaders(self.headers or {})
        url_strs_generator = GenerationCheckGenerator(url_strs)
        if self.recursion_requested:
            url_strs_generator = ConvertRecursiveToFlatWildcard(url_strs_generator)
        name_expansion_iterator = NameExpansionIterator(self.command_name, self.debug, self.logger, self.gsutil_api, url_strs_generator, self.recursion_requested, project_id=self.project_id, continue_on_error=self.continue_on_error or self.parallel_operations, bucket_listing_fields=['name', 'size'])
        seek_ahead_iterator = None
        if not self.read_args_from_stdin:
            seek_ahead_url_strs = ConvertRecursiveToFlatWildcard(url_strs)
            seek_ahead_iterator = SeekAheadNameExpansionIterator(self.command_name, self.debug, self.GetSeekAheadGsutilApi(), seek_ahead_url_strs, self.recursion_requested, all_versions=self.all_versions, project_id=self.project_id)
        for i in range(0, MAX_DECRYPTION_KEYS):
            key_number = i + 1
            keywrapper = CryptoKeyWrapperFromKey(config.get('GSUtil', 'decryption_key%s' % str(key_number), None))
            if keywrapper is None:
                break
            if keywrapper.crypto_type == CryptoKeyType.CSEK:
                self.csek_hash_to_keywrapper[keywrapper.crypto_key_sha256] = keywrapper
        if self.boto_file_encryption_sha256 is not None:
            self.csek_hash_to_keywrapper[self.boto_file_encryption_sha256] = self.boto_file_encryption_keywrapper
        if self.boto_file_encryption_keywrapper is None:
            msg = '\n'.join(textwrap.wrap('NOTE: No encryption_key was specified in the boto configuration file, so gsutil will not provide an encryption key in its rewrite API requests. This will decrypt the objects unless they are in buckets with a default KMS key set, in which case the service will automatically encrypt the rewritten objects with that key.'))
            print('%s\n' % msg, file=sys.stderr)
        self.Apply(_RewriteFuncWrapper, name_expansion_iterator, _RewriteExceptionHandler, fail_on_error=not self.continue_on_error, shared_attrs=['op_failure_count'], seek_ahead_iterator=seek_ahead_iterator)
        if self.op_failure_count:
            plural_str = 's' if self.op_failure_count else ''
            raise CommandException('%d file%s/object%s could not be rewritten.' % (self.op_failure_count, plural_str, plural_str))
        return 0

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