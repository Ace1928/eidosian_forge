from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsEnrollRequest(_messages.Message):
    """A GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsEnrollRequest
  object.

  Fields:
    enrollVmwareNodePoolRequest: A EnrollVmwareNodePoolRequest resource to be
      passed as the request body.
    parent: Required. The parent resource where the node pool is enrolled in.
  """
    enrollVmwareNodePoolRequest = _messages.MessageField('EnrollVmwareNodePoolRequest', 1)
    parent = _messages.StringField(2, required=True)