from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmmigrationProjectsLocationsSourcesMigratingVmsListRequest(_messages.Message):
    """A VmmigrationProjectsLocationsSourcesMigratingVmsListRequest object.

  Enums:
    ViewValueValuesEnum: Optional. The level of details of each migrating VM.

  Fields:
    filter: Optional. The filter request.
    orderBy: Optional. the order by fields for the result.
    pageSize: Optional. The maximum number of migrating VMs to return. The
      service may return fewer than this value. If unspecified, at most 500
      migrating VMs will be returned. The maximum value is 1000; values above
      1000 will be coerced to 1000.
    pageToken: Required. A page token, received from a previous
      `ListMigratingVms` call. Provide this to retrieve the subsequent page.
      When paginating, all other parameters provided to `ListMigratingVms`
      must match the call that provided the page token.
    parent: Required. The parent, which owns this collection of MigratingVms.
    view: Optional. The level of details of each migrating VM.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional. The level of details of each migrating VM.

    Values:
      MIGRATING_VM_VIEW_UNSPECIFIED: View is unspecified. The API will
        fallback to the default value.
      MIGRATING_VM_VIEW_BASIC: Get the migrating VM basic details. The basic
        details do not include the recent clone jobs and recent cutover jobs
        lists.
      MIGRATING_VM_VIEW_FULL: Include everything.
    """
        MIGRATING_VM_VIEW_UNSPECIFIED = 0
        MIGRATING_VM_VIEW_BASIC = 1
        MIGRATING_VM_VIEW_FULL = 2
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 6)