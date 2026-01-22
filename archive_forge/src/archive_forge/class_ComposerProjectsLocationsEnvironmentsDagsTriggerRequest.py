from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsDagsTriggerRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsDagsTriggerRequest object.

  Fields:
    dag: Required. The resource name of the DAG to trigger. Must be in the
      form: "projects/{projectId}/locations/{locationId}/environments/{environ
      mentId}/dags/{dagId}".
    triggerDagRequest: A TriggerDagRequest resource to be passed as the
      request body.
  """
    dag = _messages.StringField(1, required=True)
    triggerDagRequest = _messages.MessageField('TriggerDagRequest', 2)