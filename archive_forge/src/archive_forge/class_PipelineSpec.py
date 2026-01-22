from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PipelineSpec(_messages.Message):
    """PipelineSpec defines the desired state of Pipeline.

  Fields:
    finallyTasks: List of Tasks that execute just before leaving the Pipeline
      i.e. either after all Tasks are finished executing successfully or after
      a failure which would result in ending the Pipeline.
    generatedYaml: Output only. auto-generated yaml that is output only for
      display purpose for workflows using pipeline_spec, used by UI/gcloud cli
      for Workflows.
    params: List of parameters.
    results: Optional. Output only. List of results written out by the
      pipeline's containers
    tasks: List of Tasks that execute when this Pipeline is run.
    workspaces: Workspaces declares a set of named workspaces that are
      expected to be provided by a PipelineRun.
  """
    finallyTasks = _messages.MessageField('PipelineTask', 1, repeated=True)
    generatedYaml = _messages.StringField(2)
    params = _messages.MessageField('ParamSpec', 3, repeated=True)
    results = _messages.MessageField('PipelineResult', 4, repeated=True)
    tasks = _messages.MessageField('PipelineTask', 5, repeated=True)
    workspaces = _messages.MessageField('PipelineWorkspaceDeclaration', 6, repeated=True)