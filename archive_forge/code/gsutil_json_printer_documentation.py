from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_printer_base
Prints one element of a JSON-serializable Python object resource list.

    Allows intermingled delimit=True and delimit=False.

    Args:
      record: A JSON-serializable object.
      delimit: Dump one record if False, used by PrintSingleRecord().
    