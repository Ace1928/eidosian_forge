from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResolveServiceResponse(_messages.Message):
    """The response message for LookupService.ResolveService.

  Fields:
    service: A Service attribute.
  """
    service = _messages.MessageField('Service', 1)