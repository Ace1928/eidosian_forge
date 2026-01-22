from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchDomainsResponse(_messages.Message):
    """Response for the `SearchDomains` method.

  Fields:
    registerParameters: Results of the domain name search.
  """
    registerParameters = _messages.MessageField('RegisterParameters', 1, repeated=True)