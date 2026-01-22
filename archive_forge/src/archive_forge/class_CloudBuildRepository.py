from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudBuildRepository(_messages.Message):
    """CloudBuildRepository represents a cloud build repository.

  Fields:
    name: Required. Name of the cloud build repository. Format is
      projects/{p}/locations/{l}/connections/{c}/repositories/{r}.
    serviceAccount: Required. service_account to use for running cloud build
      triggers.
    tag: Required. tag of the cloud build repository that should be read from.
    variants: Required. variants is the configuration for how to read the
      repository to find variants.
  """
    name = _messages.StringField(1)
    serviceAccount = _messages.StringField(2)
    tag = _messages.StringField(3)
    variants = _messages.MessageField('Variants', 4)