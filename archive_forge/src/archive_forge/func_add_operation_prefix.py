from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import properties
def add_operation_prefix(job_operation_string_or_list):
    """Adds prefix to transfer operation(s) if necessary."""
    return _add_transfer_prefix(_OPERATIONS_PREFIX_REGEX, _OPERATIONS_PREFIX_STRING, job_operation_string_or_list)