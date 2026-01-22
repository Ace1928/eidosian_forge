from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDatasetsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsDatasetsGetRequest object.

  Fields:
    name: Required. The name of the Dataset resource.
    readMask: Mask specifying which fields to read.
  """
    name = _messages.StringField(1, required=True)
    readMask = _messages.StringField(2)