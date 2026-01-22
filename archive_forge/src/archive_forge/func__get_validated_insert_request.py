from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import copy
import json
from apitools.base.py import encoding_helper
from apitools.base.py import transfer
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage import retry_util
from googlecloudsdk.api_lib.storage.gcs_json import metadata_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import scaled_integer
import six
def _get_validated_insert_request(self):
    """Get an insert request that includes validated object metadata."""
    if self._request_config.predefined_acl_string:
        predefined_acl = getattr(self._messages.StorageObjectsInsertRequest.PredefinedAclValueValuesEnum, self._request_config.predefined_acl_string)
    else:
        predefined_acl = None
    object_metadata = self._messages.Object(name=self._destination_resource.storage_url.object_name, bucket=self._destination_resource.storage_url.bucket_name)
    if isinstance(self._source_resource, resource_reference.ObjectResource) and self._source_resource.custom_fields:
        object_metadata.metadata = encoding_helper.DictToAdditionalPropertyMessage(self._source_resource.custom_fields, self._messages.Object.MetadataValue)
    self._copy_acl_from_source_if_source_is_a_cloud_object_and_preserve_acl_is_true(object_metadata)
    metadata_util.update_object_metadata_from_request_config(object_metadata, self._request_config, attributes_resource=self._source_resource, posix_to_set=self._posix_to_set)
    return self._messages.StorageObjectsInsertRequest(bucket=object_metadata.bucket, object=object_metadata, ifGenerationMatch=copy_util.get_generation_match_value(self._request_config), ifMetagenerationMatch=self._request_config.precondition_metageneration_match, predefinedAcl=predefined_acl)