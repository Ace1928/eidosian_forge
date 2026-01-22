from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProducerPscConfig(_messages.Message):
    """The PSC configurations on producer side.

  Fields:
    serviceAttachmentUri: The resource path of a service attachment. Example:
      projects/{projectNumOrId}/regions/{region}/serviceAttachments/{resourceI
      d}.
  """
    serviceAttachmentUri = _messages.StringField(1)