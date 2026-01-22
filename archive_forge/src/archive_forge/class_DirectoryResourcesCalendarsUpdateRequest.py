from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryResourcesCalendarsUpdateRequest(_messages.Message):
    """A DirectoryResourcesCalendarsUpdateRequest object.

  Fields:
    calendarResource: A CalendarResource resource to be passed as the request
      body.
    calendarResourceId: The unique ID of the calendar resource to update.
    customer: The unique ID for the customer's G Suite account. As an account
      administrator, you can also use the my_customer alias to represent your
      account's customer ID.
  """
    calendarResource = _messages.MessageField('CalendarResource', 1)
    calendarResourceId = _messages.StringField(2, required=True)
    customer = _messages.StringField(3, required=True)