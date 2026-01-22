from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsRagCorporaRagFilesDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsRagCorporaRagFilesDeleteRequest object.

  Fields:
    name: Required. The name of the RagFile resource to be deleted. Format: `p
      rojects/{project}/locations/{location}/ragCorpora/{rag_corpus}/ragFiles/
      {rag_file}`
  """
    name = _messages.StringField(1, required=True)