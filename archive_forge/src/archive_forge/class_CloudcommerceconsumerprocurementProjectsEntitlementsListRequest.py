from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementProjectsEntitlementsListRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementProjectsEntitlementsListRequest
  object.

  Fields:
    filter: Filter that can be used to limit the list request. A query string
      that can match a selected set of attributes with string values.
      Supported query attributes are * `services.service_name` * `service` *
      `offer` * `pending_change.new_offer` * `product_external_name` *
      `provider` Service queries have the format: `service="services/%s"`
      where %s is the OnePlatformServiceId and all values are surrounded with
      quote literals. Offer has the format: "billingAccounts/{billing-account-
      id}/offers/{offer-id}" for private offers or
      "services/{service}/standardOffers/{offer-id}" for standard offers.
      Related offer filters are formatted where %s is the above fully
      qualified Offer and all values are surrounded with quote literals. Ex.
      `offer="%s"` `pending_change.new_offer="%s"` Product and provider
      queries have the format: `product_external_name="pumpkin-saas"`
      `provider="pumpkindb"` If the query contains special characters other
      than letters, underscore, or digits, the phrase must be quoted with
      double quotes. For example, `service="services/%s"`, where the service
      query needs to be quoted because it contains special character forward
      slash. Queries can be combined with `OR`, and `NOT` to form more complex
      queries. You can also group them to force a desired evaluation order.
      E.g. `service="services/pumpkin"`.
    pageSize: The maximum number of entries requested. The default page size
      is 25 and the maximum page size is 200.
    pageToken: The token for fetching the next page.
    parent: Required. The parent resource to query for Entitlements. Currently
      the only parents supported are "projects/{project-number}" and
      "projects/{project-id}".
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)