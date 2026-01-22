from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsModelsPatchRequest(_messages.Message):
    """A MlProjectsModelsPatchRequest object.

  Fields:
    googleCloudMlV1Model: A GoogleCloudMlV1Model resource to be passed as the
      request body.
    name: Required. The project name.
    updateMask: Required. Specifies the path, relative to `Model`, of the
      field to update. For example, to change the description of a model to
      "foo" and set its default version to "version_1", the `update_mask`
      parameter would be specified as `description`, `default_version.name`,
      and the `PATCH` request body would specify the new value, as follows: {
      "description": "foo", "defaultVersion": { "name":"version_1" } }
      Currently the supported update masks are `description` and
      `default_version.name`.
  """
    googleCloudMlV1Model = _messages.MessageField('GoogleCloudMlV1Model', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)