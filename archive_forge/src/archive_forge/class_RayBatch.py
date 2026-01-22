from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RayBatch(_messages.Message):
    """A configuration for running an Ray Job
  (https://docs.ray.io/en/latest/cluster/running-applications/job-
  submission/index.html) workload.

  Fields:
    args: Optional. The arguments to pass to the Ray job script.
    mainPythonFileUri: Required. The HCFS URI of the main Python file to use
      as the Ray job. Must be a .py file.
  """
    args = _messages.StringField(1, repeated=True)
    mainPythonFileUri = _messages.StringField(2)