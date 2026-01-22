from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsDagsGetSourceCodeRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsDagsGetSourceCodeRequest object.

  Fields:
    dag: Required. The resource name of the DAG to fetch source code of. Must
      be in the form: "projects/{projectId}/locations/{locationId}/environment
      s/{environmentId}/dags/{dagId}".
  """
    dag = _messages.StringField(1, required=True)