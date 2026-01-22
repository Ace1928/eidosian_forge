from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1beta1ExportDocumentsResponse(_messages.Message):
    """Returned in the google.longrunning.Operation response field.

  Fields:
    outputUriPrefix: Location of the output files. This can be used to begin
      an import into Cloud Firestore (this project or another project) after
      the operation completes successfully.
  """
    outputUriPrefix = _messages.StringField(1)