from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.command_lib.storage.tasks import task
Initializes task.

    Args:
      source_resource (resource_reference.Resource): Source resource to copy.
      destination_resource (resource_reference.Resource): Target resource to
        copy to.
      offset (int): The index of the first byte in the range.
      length (int): The number of bytes in the range.
      component_number (int): If a multipart operation, indicates the
        component number.
      total_components (int): If a multipart operation, indicates the
        total number of components.
    