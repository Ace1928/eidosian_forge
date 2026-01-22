from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import properties
def add_job_prefix(job_name_string_or_list):
    """Adds prefix to transfer job(s) if necessary."""
    return _add_transfer_prefix(_JOBS_PREFIX_REGEX, _JOBS_PREFIX_STRING, job_name_string_or_list)