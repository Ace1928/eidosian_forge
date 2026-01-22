from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisVersionsCreateRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisVersionsCreateRequest object.

  Fields:
    apiVersion: A ApiVersion resource to be passed as the request body.
    apiVersionId: Required. The ID to use for the version, which will become
      the final component of the version's resource name. This value should be
      1-63 characters, and valid characters are /a-z-/. Following AIP-162, IDs
      must not have the form of a UUID.
    parent: Required. The parent, which owns this collection of versions.
      Format: `projects/*/locations/*/apis/*`
  """
    apiVersion = _messages.MessageField('ApiVersion', 1)
    apiVersionId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)