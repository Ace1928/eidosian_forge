from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListCaPoolsResponse(_messages.Message):
    """Response message for CertificateAuthorityService.ListCaPools.

  Fields:
    caPools: The list of CaPools.
    nextPageToken: A token to retrieve next page of results. Pass this value
      in ListCertificateAuthoritiesRequest.next_page_token to retrieve the
      next page of results.
    unreachable: A list of locations (e.g. "us-west1") that could not be
      reached.
  """
    caPools = _messages.MessageField('CaPool', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)