from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GraphqlError(_messages.Message):
    """GraphqlError conforms to the GraphQL error spec.
  https://spec.graphql.org/draft/#sec-Errors Firebase Data Connect API
  surfaces `GraphqlError` in various APIs: - Upon compile error,
  `UpdateSchema` and `UpdateConnector` return Code.Invalid_Argument with a
  list of `GraphqlError` in error details. - Upon query compile error,
  `ExecuteGraphql` and `ExecuteGraphqlRead` return Code.OK with a list of
  `GraphqlError` in response body. - Upon query execution error,
  `ExecuteGraphql`, `ExecuteGraphqlRead`, `ExecuteMutation` and `ExecuteQuery`
  all return Code.OK with a list of `GraphqlError` in response body.

  Fields:
    extensions: Additional error information.
    locations: The source locations where the error occurred. Locations should
      help developers and toolings identify the source of error quickly.
      Included in admin endpoints (`ExecuteGraphql`, `ExecuteGraphqlRead`,
      `UpdateSchema` and `UpdateConnector`) to reference the provided GraphQL
      GQL document. Omitted in `ExecuteMutation` and `ExecuteQuery` since the
      caller shouldn't have access access the underlying GQL source.
    message: The detailed error message. The message should help developer
      understand the underlying problem without leaking internal data.
    path: The result field which could not be populated due to error. Clients
      can use path to identify whether a null result is intentional or caused
      by a runtime error. It should be a list of string or index from the root
      of GraphQL query document.
  """
    extensions = _messages.MessageField('GraphqlErrorExtensions', 1)
    locations = _messages.MessageField('SourceLocation', 2, repeated=True)
    message = _messages.StringField(3)
    path = _messages.MessageField('extra_types.JsonValue', 4, repeated=True)