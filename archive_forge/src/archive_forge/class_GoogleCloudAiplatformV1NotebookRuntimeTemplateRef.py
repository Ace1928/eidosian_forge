from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1NotebookRuntimeTemplateRef(_messages.Message):
    """Points to a NotebookRuntimeTemplateRef.

  Fields:
    notebookRuntimeTemplate: Immutable. A resource name of the
      NotebookRuntimeTemplate.
  """
    notebookRuntimeTemplate = _messages.StringField(1)