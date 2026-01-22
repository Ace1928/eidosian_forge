from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsApiproductsRateplansListRequest(_messages.Message):
    """A ApigeeOrganizationsApiproductsRateplansListRequest object.

  Enums:
    StateValueValuesEnum: State of the rate plans (`DRAFT`, `PUBLISHED`) that
      you want to display.

  Fields:
    count: Number of rate plans to return in the API call. Use with the
      `startKey` parameter to provide more targeted filtering. The maximum
      limit is 1000. Defaults to 100.
    expand: Flag that specifies whether to expand the results. Set to `true`
      to get expanded details about each API. Defaults to `false`.
    orderBy: Name of the attribute used for sorting. Valid values include: *
      `name`: Name of the rate plan. * `state`: State of the rate plan
      (`DRAFT`, `PUBLISHED`). * `startTime`: Time when the rate plan becomes
      active. * `endTime`: Time when the rate plan expires. **Note**: Not
      supported by Apigee at this time.
    parent: Required. Name of the API product. Use the following structure in
      your request: `organizations/{org}/apiproducts/{apiproduct}` Use
      `organizations/{org}/apiproducts/-` to return rate plans for all API
      products within the organization.
    startKey: Name of the rate plan from which to start displaying the list of
      rate plans. If omitted, the list starts from the first item. For
      example, to view the rate plans from 51-150, set the value of `startKey`
      to the name of the 51st rate plan and set the value of `count` to 100.
    state: State of the rate plans (`DRAFT`, `PUBLISHED`) that you want to
      display.
  """

    class StateValueValuesEnum(_messages.Enum):
        """State of the rate plans (`DRAFT`, `PUBLISHED`) that you want to
    display.

    Values:
      STATE_UNSPECIFIED: State of the rate plan is not specified.
      DRAFT: Rate plan is in draft mode and only visible to API providers.
      PUBLISHED: Rate plan is published and will become visible to developers
        for the configured duration (between `startTime` and `endTime`).
    """
        STATE_UNSPECIFIED = 0
        DRAFT = 1
        PUBLISHED = 2
    count = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    expand = _messages.BooleanField(2)
    orderBy = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    startKey = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)