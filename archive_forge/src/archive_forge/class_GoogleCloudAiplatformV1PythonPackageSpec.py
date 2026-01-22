from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1PythonPackageSpec(_messages.Message):
    """The spec of a Python packaged code.

  Fields:
    args: Command line arguments to be passed to the Python task.
    env: Environment variables to be passed to the python module. Maximum
      limit is 100.
    executorImageUri: Required. The URI of a container image in Artifact
      Registry that will run the provided Python package. Vertex AI provides a
      wide range of executor images with pre-installed packages to meet users'
      various use cases. See the list of [pre-built containers for
      training](https://cloud.google.com/vertex-ai/docs/training/pre-built-
      containers). You must use an image from this list.
    packageUris: Required. The Google Cloud Storage location of the Python
      package files which are the training program and its dependent packages.
      The maximum number of package URIs is 100.
    pythonModule: Required. The Python module name to run after installing the
      packages.
  """
    args = _messages.StringField(1, repeated=True)
    env = _messages.MessageField('GoogleCloudAiplatformV1EnvVar', 2, repeated=True)
    executorImageUri = _messages.StringField(3)
    packageUris = _messages.StringField(4, repeated=True)
    pythonModule = _messages.StringField(5)