from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import math
import os
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import scaled_integer
def _get_temporary_component_name(source_resource, destination_resource, random_prefix, component_id):
    """Gets a temporary object name for a component of source_resource."""
    source_name = source_resource.storage_url.object_name
    salted_name = _PARALLEL_UPLOAD_STATIC_PREFIX + source_name
    sha1_hash = hashlib.sha1(salted_name.encode('utf-8'))
    component_prefix = properties.VALUES.storage.parallel_composite_upload_component_prefix.Get()
    delimiter = destination_resource.storage_url.delimiter
    if component_prefix.startswith(delimiter):
        prefix = component_prefix.lstrip(delimiter)
    else:
        destination_object_name = destination_resource.storage_url.object_name
        destination_prefix, _, _ = destination_object_name.rpartition(delimiter)
        prefix = _ensure_truthy_path_ends_with_single_delimiter(destination_prefix, delimiter) + component_prefix
    return '{}{}_{}_{}'.format(_ensure_truthy_path_ends_with_single_delimiter(prefix, delimiter), random_prefix, sha1_hash.hexdigest(), str(component_id))