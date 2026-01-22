from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsGlobalFeaturesPatchRequest(_messages.Message):
    """A GkehubProjectsLocationsGlobalFeaturesPatchRequest object.

  Fields:
    feature: A Feature resource to be passed as the request body.
    name: Required. The Feature resource name in the format
      `projects/*/locations/global/features/*`.
    updateMask: Mask of fields to update.
  """
    feature = _messages.MessageField('Feature', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)