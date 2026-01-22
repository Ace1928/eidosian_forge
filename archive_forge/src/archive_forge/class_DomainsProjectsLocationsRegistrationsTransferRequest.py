from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainsProjectsLocationsRegistrationsTransferRequest(_messages.Message):
    """A DomainsProjectsLocationsRegistrationsTransferRequest object.

  Fields:
    parent: Required. The parent resource of the `Registration`. Must be in
      the format `projects/*/locations/*`.
    transferDomainRequest: A TransferDomainRequest resource to be passed as
      the request body.
  """
    parent = _messages.StringField(1, required=True)
    transferDomainRequest = _messages.MessageField('TransferDomainRequest', 2)