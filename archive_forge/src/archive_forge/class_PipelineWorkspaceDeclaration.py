from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PipelineWorkspaceDeclaration(_messages.Message):
    """Workspaces declares a set of named workspaces that are expected to be
  provided by a PipelineRun.

  Fields:
    description: Description is a human readable string describing how the
      workspace will be used in the Pipeline.
    name: Name is the name of a workspace to be provided by a PipelineRun.
    optional: Optional marks a Workspace as not being required in
      PipelineRuns. By default this field is false and so declared workspaces
      are required.
  """
    description = _messages.StringField(1)
    name = _messages.StringField(2)
    optional = _messages.BooleanField(3)