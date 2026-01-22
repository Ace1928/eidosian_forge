from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExtractPostureRequest(_messages.Message):
    """Message for extracting existing policies on a workload as a Posture.

  Fields:
    postureId: Required. User provided identifier. It should be unique in
      scope of an Organization and location.
    workload: Required. Workload from which the policies are to be extracted,
      it should belong to the same organization defined in parent. The format
      of this value varies depending on the scope of the request: -
      `folder/folderNumber` - `project/projectNumber` -
      `organization/organizationNumber`
  """
    postureId = _messages.StringField(1)
    workload = _messages.StringField(2)