from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EssentialcontactsFoldersContactsComputeRequest(_messages.Message):
    """A EssentialcontactsFoldersContactsComputeRequest object.

  Enums:
    NotificationCategoriesValueValuesEnum: The categories of notifications to
      compute contacts for. If ALL is included in this list, contacts
      subscribed to any notification category will be returned.

  Fields:
    notificationCategories: The categories of notifications to compute
      contacts for. If ALL is included in this list, contacts subscribed to
      any notification category will be returned.
    pageSize: Optional. The maximum number of results to return from this
      request. Non-positive values are ignored. The presence of
      `next_page_token` in the response indicates that more results might be
      available. If not specified, the default page_size is 100.
    pageToken: Optional. If present, retrieves the next batch of results from
      the preceding call to this method. `page_token` must be the value of
      `next_page_token` from the previous response. The values of other method
      parameters should be identical to those in the previous call.
    parent: Required. The name of the resource to compute contacts for.
      Format: organizations/{organization_id}, folders/{folder_id} or
      projects/{project_id}
  """

    class NotificationCategoriesValueValuesEnum(_messages.Enum):
        """The categories of notifications to compute contacts for. If ALL is
    included in this list, contacts subscribed to any notification category
    will be returned.

    Values:
      NOTIFICATION_CATEGORY_UNSPECIFIED: Notification category is unrecognized
        or unspecified.
      ALL: All notifications related to the resource, including notifications
        pertaining to categories added in the future.
      SUSPENSION: Notifications related to imminent account suspension.
      SECURITY: Notifications related to security/privacy incidents,
        notifications, and vulnerabilities.
      TECHNICAL: Notifications related to technical events and issues such as
        outages, errors, or bugs.
      BILLING: Notifications related to billing and payments notifications,
        price updates, errors, or credits.
      LEGAL: Notifications related to enforcement actions, regulatory
        compliance, or government notices.
      PRODUCT_UPDATES: Notifications related to new versions, product terms
        updates, or deprecations.
      TECHNICAL_INCIDENTS: Child category of TECHNICAL. If assigned, technical
        incident notifications will go to these contacts instead of TECHNICAL.
    """
        NOTIFICATION_CATEGORY_UNSPECIFIED = 0
        ALL = 1
        SUSPENSION = 2
        SECURITY = 3
        TECHNICAL = 4
        BILLING = 5
        LEGAL = 6
        PRODUCT_UPDATES = 7
        TECHNICAL_INCIDENTS = 8
    notificationCategories = _messages.EnumField('NotificationCategoriesValueValuesEnum', 1, repeated=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)