from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import fnmatch
import heapq
import os
import pathlib
import re
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as api_errors
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
import six
def _decrypt_resource_if_necessary(self, resource):
    if self._fetch_encrypted_object_hashes and cloud_api.Capability.ENCRYPTION in self._client.capabilities and (self._fields_scope != cloud_api.FieldsScope.SHORT) and isinstance(resource, resource_reference.ObjectResource) and (not (resource.crc32c_hash or resource.md5_hash)):
        if resource.kms_key:
            return self._client.get_object_metadata(resource.bucket, resource.name, generation=self._url.generation, fields_scope=self._fields_scope, soft_deleted=self._soft_deleted)
        if resource.decryption_key_hash_sha256:
            request_config = request_config_factory.get_request_config(resource.storage_url, decryption_key_hash_sha256=resource.decryption_key_hash_sha256, error_on_missing_key=self._error_on_missing_key)
            if getattr(request_config.resource_args, 'decryption_key', None):
                return self._client.get_object_metadata(resource.bucket, resource.name, request_config, generation=self._url.generation, fields_scope=self._fields_scope, soft_deleted=self._soft_deleted)
    return resource