from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsRestorePlansCreateRequest(_messages.Message):
    """A GkebackupProjectsLocationsRestorePlansCreateRequest object.

  Fields:
    parent: Required. The location within which to create the RestorePlan.
      Format: `projects/*/locations/*`
    restorePlan: A RestorePlan resource to be passed as the request body.
    restorePlanId: Required. The client-provided short name for the
      RestorePlan resource. This name must: - be between 1 and 63 characters
      long (inclusive) - consist of only lower-case ASCII letters, numbers,
      and dashes - start with a lower-case letter - end with a lower-case
      letter or number - be unique within the set of RestorePlans in this
      location
  """
    parent = _messages.StringField(1, required=True)
    restorePlan = _messages.MessageField('RestorePlan', 2)
    restorePlanId = _messages.StringField(3)