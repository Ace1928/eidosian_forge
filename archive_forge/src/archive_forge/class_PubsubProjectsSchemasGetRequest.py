from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSchemasGetRequest(_messages.Message):
    """A PubsubProjectsSchemasGetRequest object.

  Enums:
    ViewValueValuesEnum: The set of fields to return in the response. If not
      set, returns a Schema with all fields filled out. Set to `BASIC` to omit
      the `definition`.

  Fields:
    name: Required. The name of the schema to get. Format is
      `projects/{project}/schemas/{schema}`.
    view: The set of fields to return in the response. If not set, returns a
      Schema with all fields filled out. Set to `BASIC` to omit the
      `definition`.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """The set of fields to return in the response. If not set, returns a
    Schema with all fields filled out. Set to `BASIC` to omit the
    `definition`.

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
    view = _messages.EnumField('ViewValueValuesEnum', 2)