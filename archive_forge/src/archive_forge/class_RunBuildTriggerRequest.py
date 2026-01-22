from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunBuildTriggerRequest(_messages.Message):
    """Specifies a build trigger to run and the source to use.

  Fields:
    projectId: Required. ID of the project.
    source: Source to build against this trigger. Branch and tag names cannot
      consist of regular expressions.
    triggerId: Required. ID of the trigger.
  """
    projectId = _messages.StringField(1)
    source = _messages.MessageField('RepoSource', 2)
    triggerId = _messages.StringField(3)