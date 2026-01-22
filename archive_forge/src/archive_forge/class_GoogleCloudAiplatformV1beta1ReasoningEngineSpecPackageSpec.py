from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ReasoningEngineSpecPackageSpec(_messages.Message):
    """User provided package spec like pickled object and package requirements.

  Fields:
    dependencyFilesGcsUri: Optional. The Cloud Storage URI of the dependency
      files in tar.gz format.
    pickleObjectGcsUri: Optional. The Cloud Storage URI of the pickled python
      object.
    pythonVersion: Optional. The Python version. Currently support 3.8, 3.9,
      3.10, 3.11. If not specified, default value is 3.10.
    requirementsGcsUri: Optional. The Cloud Storage URI of the
      `requirements.txt` file
  """
    dependencyFilesGcsUri = _messages.StringField(1)
    pickleObjectGcsUri = _messages.StringField(2)
    pythonVersion = _messages.StringField(3)
    requirementsGcsUri = _messages.StringField(4)