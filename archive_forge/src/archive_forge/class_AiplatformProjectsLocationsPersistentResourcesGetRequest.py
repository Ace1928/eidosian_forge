from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsPersistentResourcesGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsPersistentResourcesGetRequest object.

  Fields:
    name: Required. The name of the PersistentResource resource. Format: `proj
      ects/{project_id_or_number}/locations/{location_id}/persistentResources/
      {persistent_resource_id}`
  """
    name = _messages.StringField(1, required=True)