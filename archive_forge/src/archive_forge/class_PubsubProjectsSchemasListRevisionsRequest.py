from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSchemasListRevisionsRequest(_messages.Message):
    """A PubsubProjectsSchemasListRevisionsRequest object.

  Enums:
    ViewValueValuesEnum: The set of Schema fields to return in the response.
      If not set, returns Schemas with `name` and `type`, but not
      `definition`. Set to `FULL` to retrieve all fields.

  Fields:
    name: Required. The name of the schema to list revisions for.
    pageSize: The maximum number of revisions to return per page.
    pageToken: The page token, received from a previous ListSchemaRevisions
      call. Provide this to retrieve the subsequent page.
    view: The set of Schema fields to return in the response. If not set,
      returns Schemas with `name` and `type`, but not `definition`. Set to
      `FULL` to retrieve all fields.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """The set of Schema fields to return in the response. If not set,
    returns Schemas with `name` and `type`, but not `definition`. Set to
    `FULL` to retrieve all fields.

    Values:
      SCHEMA_VIEW_UNSPECIFIED: The default / unset value. The API will default
        to the BASIC view.
      BASIC: Include the name and type of the schema, but not the definition.
      FULL: Include all Schema object fields.
    """
        SCHEMA_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    view = _messages.EnumField('ViewValueValuesEnum', 4)