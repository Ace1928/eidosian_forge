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
Calculates start bytes and sizes for a multi-component copy operation.

  Args:
    file_size (int): Total byte size of file being divided into components.
    component_count (int): Number of components to divide file into.

  Returns:
    List of component offsets and lengths: list[(offset, length)].
    Total component count can be found by taking the length of the list.
  