from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsLocationsStudiesTrialsListRequest(_messages.Message):
    """A MlProjectsLocationsStudiesTrialsListRequest object.

  Fields:
    parent: Required. The name of the study that the trial belongs to.
  """
    parent = _messages.StringField(1, required=True)