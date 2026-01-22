from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RequesterJustificationConfig(_messages.Message):
    """Defines the ways in which a requester should provide the justification
  while requesting for access.

  Fields:
    notMandatory: This option means the requester will not be forced to
      provide a justification.
    unstructured: This option means the requester has to necessarily provide a
      free flowing text as justification. If this is selected the server will
      allow the requester to provide a justification but will not validate it.
  """
    notMandatory = _messages.MessageField('NotMandatory', 1)
    unstructured = _messages.MessageField('Unstructured', 2)