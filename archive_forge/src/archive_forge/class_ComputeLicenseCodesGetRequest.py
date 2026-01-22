from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeLicenseCodesGetRequest(_messages.Message):
    """A ComputeLicenseCodesGetRequest object.

  Fields:
    licenseCode: Number corresponding to the License code resource to return.
    project: Project ID for this request.
  """
    licenseCode = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)