from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresConsentsListRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsConsentStoresConsentsListRequest
  object.

  Fields:
    filter: Optional. Restricts the consents returned to those matching a
      filter. The following syntax is available: * A string field value can be
      written as text inside quotation marks, for example `"query text"`. The
      only valid relational operation for text fields is equality (`=`), where
      text is searched within the field, rather than having the field be equal
      to the text. For example, `"Comment = great"` returns messages with
      `great` in the comment field. * A number field value can be written as
      an integer, a decimal, or an exponential. The valid relational operators
      for number fields are the equality operator (`=`), along with the less
      than/greater than operators (`<`, `<=`, `>`, `>=`). Note that there is
      no inequality (`!=`) operator. You can prepend the `NOT` operator to an
      expression to negate it. * A date field value must be written in `yyyy-
      mm-dd` form. Fields with date and time use the RFC3339 time format.
      Leading zeros are required for one-digit months and days. The valid
      relational operators for date fields are the equality operator (`=`) ,
      along with the less than/greater than operators (`<`, `<=`, `>`, `>=`).
      Note that there is no inequality (`!=`) operator. You can prepend the
      `NOT` operator to an expression to negate it. * Multiple field query
      expressions can be combined in one query by adding `AND` or `OR`
      operators between the expressions. If a boolean operator appears within
      a quoted string, it is not treated as special, it's just another part of
      the character string to be matched. You can prepend the `NOT` operator
      to an expression to negate it. The fields available for filtering are: -
      user_id. For example, `filter='user_id="user123"'`. - consent_artifact -
      state - revision_create_time - expire_time - metadata. For example,
      `filter=Metadata(\\"testkey\\")=\\"value\\"` or
      `filter=HasMetadata(\\"testkey\\")`.
    pageSize: Optional. Limit on the number of Consents to return in a single
      response. If not specified, 100 is used. May not be larger than 1000.
    pageToken: Optional. The next_page_token value returned from the previous
      List request, if any.
    parent: Required. Name of the consent store to retrieve Consents from.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)