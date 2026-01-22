from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LifesciencesProjectsLocationsPipelinesRunRequest(_messages.Message):
    """A LifesciencesProjectsLocationsPipelinesRunRequest object.

  Fields:
    parent: The project and location that this request should be executed
      against.
    runPipelineRequest: A RunPipelineRequest resource to be passed as the
      request body.
  """
    parent = _messages.StringField(1, required=True)
    runPipelineRequest = _messages.MessageField('RunPipelineRequest', 2)