from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsGlobalFeaturesDeleteRequest(_messages.Message):
    """A GkehubProjectsLocationsGlobalFeaturesDeleteRequest object.

  Fields:
    force: If set to true, the delete will ignore any outstanding resources
      for this Feature (that is, `FeatureState.has_resources` is set to true).
      These resources will NOT be cleaned up or modified in any way.
    name: Required. The Feature resource name in the format
      `projects/*/locations/global/features/*`.
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)