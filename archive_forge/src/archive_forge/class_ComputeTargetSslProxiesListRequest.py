from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeTargetSslProxiesListRequest(_messages.Message):
    """A ComputeTargetSslProxiesListRequest object.

  Fields:
    filter: A filter expression that filters resources listed in the response.
      Most Compute resources support two types of filter expressions:
      expressions that support regular expressions and expressions that follow
      API improvement proposal AIP-160. These two types of filter expressions
      cannot be mixed in one request. If you want to use AIP-160, your
      expression must specify the field name, an operator, and the value that
      you want to use for filtering. The value must be a string, a number, or
      a boolean. The operator must be either `=`, `!=`, `>`, `<`, `<=`, `>=`
      or `:`. For example, if you are filtering Compute Engine instances, you
      can exclude instances named `example-instance` by specifying `name !=
      example-instance`. The `:*` comparison can be used to test whether a key
      has been defined. For example, to find all objects with `owner` label
      use: ``` labels.owner:* ``` You can also filter nested fields. For
      example, you could specify `scheduling.automaticRestart = false` to
      include instances only if they are not scheduled for automatic restarts.
      You can use filtering on nested fields to filter based on resource
      labels. To filter on multiple expressions, provide each separate
      expression within parentheses. For example: ```
      (scheduling.automaticRestart = true) (cpuPlatform = "Intel Skylake") ```
      By default, each expression is an `AND` expression. However, you can
      include `AND` and `OR` expressions explicitly. For example: ```
      (cpuPlatform = "Intel Skylake") OR (cpuPlatform = "Intel Broadwell") AND
      (scheduling.automaticRestart = true) ``` If you want to use a regular
      expression, use the `eq` (equal) or `ne` (not equal) operator against a
      single un-parenthesized expression with or without quotes or against
      multiple parenthesized expressions. Examples: `fieldname eq unquoted
      literal` `fieldname eq 'single quoted literal'` `fieldname eq "double
      quoted literal"` `(fieldname1 eq literal) (fieldname2 ne "literal")` The
      literal value is interpreted as a regular expression using Google RE2
      library syntax. The literal value must match the entire field. For
      example, to filter for instances that do not end with name "instance",
      you would use `name ne .*instance`. You cannot combine constraints on
      multiple fields using regular expressions.
    maxResults: The maximum number of results per page that should be
      returned. If the number of available results is larger than
      `maxResults`, Compute Engine returns a `nextPageToken` that can be used
      to get the next page of results in subsequent list requests. Acceptable
      values are `0` to `500`, inclusive. (Default: `500`)
    orderBy: Sorts list results by a certain order. By default, results are
      returned in alphanumerical order based on the resource name. You can
      also sort results in descending order based on the creation timestamp
      using `orderBy="creationTimestamp desc"`. This sorts results based on
      the `creationTimestamp` field in reverse chronological order (newest
      result first). Use this to sort resources like operations so that the
      newest operation is returned first. Currently, only sorting by `name` or
      `creationTimestamp desc` is supported.
    pageToken: Specifies a page token to use. Set `pageToken` to the
      `nextPageToken` returned by a previous list request to get the next page
      of results.
    project: Project ID for this request.
    returnPartialSuccess: Opt-in for partial success behavior which provides
      partial results in case of failure. The default value is false. For
      example, when partial success behavior is enabled, aggregatedList for a
      single zone scope either returns all resources in the zone or no
      resources, with an error code.
  """
    filter = _messages.StringField(1)
    maxResults = _messages.IntegerField(2, variant=_messages.Variant.UINT32, default=500)
    orderBy = _messages.StringField(3)
    pageToken = _messages.StringField(4)
    project = _messages.StringField(5, required=True)
    returnPartialSuccess = _messages.BooleanField(6)