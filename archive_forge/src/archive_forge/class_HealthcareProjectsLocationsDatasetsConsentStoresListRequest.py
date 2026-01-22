from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresListRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsConsentStoresListRequest object.

  Fields:
    filter: Optional. Restricts the stores returned to those matching a
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
      to an expression to negate it. Only filtering on labels is supported.
      For example, `filter=labels.key=value`.
    pageSize: Optional. Limit on the number of consent stores to return in a
      single response. If not specified, 100 is used. May not be larger than
      1000.
    pageToken: Optional. Token to retrieve the next page of results, or empty
      to get the first page.
    parent: Required. Name of the dataset.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)