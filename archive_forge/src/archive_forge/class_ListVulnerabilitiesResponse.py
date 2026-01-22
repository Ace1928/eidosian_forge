from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListVulnerabilitiesResponse(_messages.Message):
    """ListVulnerabilitiesResponse contains a single page of vulnerabilities
  resulting from a scan.

  Fields:
    nextPageToken: A page token that can be used in a subsequent call to
      ListVulnerabilities to continue retrieving results.
    occurrences: The list of Vulnerability Occurrences resulting from a scan.
  """
    nextPageToken = _messages.StringField(1)
    occurrences = _messages.MessageField('Occurrence', 2, repeated=True)