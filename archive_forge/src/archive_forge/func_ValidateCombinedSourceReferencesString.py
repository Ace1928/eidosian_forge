from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
def ValidateCombinedSourceReferencesString(source_refs):
    """Determines if `source_refs` contains a valid list of source references.

  Args:
    source_refs: A multi-line string containing one source reference per line.

  Raises:
    ValidationError: If the reference is malformed.
  """
    if len(source_refs) > SOURCE_REFERENCES_MAX_SIZE:
        raise validation.ValidationError('Total source reference(s) size exceeds the limit: %d > %d' % (len(source_refs), SOURCE_REFERENCES_MAX_SIZE))
    for ref in source_refs.splitlines():
        ValidateSourceReference(ref.strip())