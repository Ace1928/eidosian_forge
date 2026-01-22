from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OsconfigProjectsLocationsInstancesInventoriesGetRequest(_messages.Message):
    """A OsconfigProjectsLocationsInstancesInventoriesGetRequest object.

  Enums:
    ViewValueValuesEnum: Inventory view indicating what information should be
      included in the inventory resource. If unspecified, the default view is
      BASIC.

  Fields:
    name: Required. API resource name for inventory resource. Format:
      `projects/{project}/locations/{location}/instances/{instance}/inventory`
      For `{project}`, either `project-number` or `project-id` can be
      provided. For `{instance}`, either Compute Engine `instance-id` or
      `instance-name` can be provided.
    view: Inventory view indicating what information should be included in the
      inventory resource. If unspecified, the default view is BASIC.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Inventory view indicating what information should be included in the
    inventory resource. If unspecified, the default view is BASIC.

    Values:
      INVENTORY_VIEW_UNSPECIFIED: The default value. The API defaults to the
        BASIC view.
      BASIC: Returns the basic inventory information that includes `os_info`.
      FULL: Returns all fields.
    """
        INVENTORY_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)