from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import six
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def IsCustomMetadataHeader(header):
    """Returns true if header (which must be lowercase) is a custom header."""
    return header.startswith('x-goog-meta-') or header.startswith('x-amz-meta-')