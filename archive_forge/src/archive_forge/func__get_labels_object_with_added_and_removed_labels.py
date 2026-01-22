from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import encoding
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.storage import metadata_util
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.api_lib.storage.gcs_json import metadata_field_converters
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import gzip_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import gcs_resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import properties
def _get_labels_object_with_added_and_removed_labels(labels_object, resource_args):
    """Returns shallow copy of bucket labels object with requested changes.

  Args:
    labels_object (messages.Bucket.LabelsValue|None): Existing labels.
    resource_args (request_config_factory._BucketConfig): Contains desired
      changes for labels list.

  Returns:
    messages.Bucket.LabelsValue|None: Contains shallow copy of labels list with
      added and removed values or None if there was no original object.
  """
    messages = apis.GetMessagesModule('storage', 'v1')
    if labels_object:
        existing_labels = labels_object.additionalProperties
    else:
        existing_labels = []
    new_labels = []
    labels_to_remove = set(resource_args.labels_to_remove or [])
    for existing_label in existing_labels:
        if existing_label.key in labels_to_remove:
            new_labels.append(messages.Bucket.LabelsValue.AdditionalProperty(key=existing_label.key, value=None))
        else:
            new_labels.append(existing_label)
    labels_to_append = resource_args.labels_to_append or {}
    for key, value in labels_to_append.items():
        new_labels.append(messages.Bucket.LabelsValue.AdditionalProperty(key=key, value=value))
    if not (labels_object or new_labels):
        return None
    return messages.Bucket.LabelsValue(additionalProperties=new_labels)