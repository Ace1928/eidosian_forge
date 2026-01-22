from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1BuiltInAlgorithmOutput(_messages.Message):
    """Represents output related to a built-in algorithm Job.

  Fields:
    framework: Framework on which the built-in algorithm was trained.
    modelPath: The Cloud Storage path to the `model/` directory where the
      training job saves the trained model. Only set for successful jobs that
      don't use hyperparameter tuning.
    pythonVersion: Python version on which the built-in algorithm was trained.
    runtimeVersion: AI Platform runtime version on which the built-in
      algorithm was trained.
  """
    framework = _messages.StringField(1)
    modelPath = _messages.StringField(2)
    pythonVersion = _messages.StringField(3)
    runtimeVersion = _messages.StringField(4)