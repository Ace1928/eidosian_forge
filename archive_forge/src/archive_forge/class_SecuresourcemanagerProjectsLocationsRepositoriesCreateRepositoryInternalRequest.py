from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuresourcemanagerProjectsLocationsRepositoriesCreateRepositoryInternalRequest(_messages.Message):
    """A SecuresourcemanagerProjectsLocationsRepositoriesCreateRepositoryIntern
  alRequest object.

  Fields:
    parent: Required. The project in which to create the repository. Values
      are of the form `projects/{project_number}/locations/{location_id}`
    repository: A Repository resource to be passed as the request body.
    repositoryId: Required. The ID to use for the repository, which will
      become the final component of the repository's resource name. This value
      should be 4-63 characters, and valid characters are /a-z-/.
  """
    parent = _messages.StringField(1, required=True)
    repository = _messages.MessageField('Repository', 2)
    repositoryId = _messages.StringField(3)