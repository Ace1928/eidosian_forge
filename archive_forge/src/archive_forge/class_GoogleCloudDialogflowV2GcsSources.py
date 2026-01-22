from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2GcsSources(_messages.Message):
    """Google Cloud Storage location for the inputs.

  Fields:
    uris: Required. Google Cloud Storage URIs for the inputs. A URI is of the
      form: `gs://bucket/object-prefix-or-name` Whether a prefix or name is
      used depends on the use case.
  """
    uris = _messages.StringField(1, repeated=True)