from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GqlQuery(_messages.Message):
    """A [GQL
  query](https://cloud.google.com/datastore/docs/apis/gql/gql_reference).

  Messages:
    NamedBindingsValue: For each non-reserved named binding site in the query
      string, there must be a named parameter with that name, but not
      necessarily the inverse. Key must match regex `A-Za-z_$*`, must not
      match regex `__.*__`, and must not be `""`.

  Fields:
    allowLiterals: When false, the query string must not contain any literals
      and instead must bind all values. For example, `SELECT * FROM Kind WHERE
      a = 'string literal'` is not allowed, while `SELECT * FROM Kind WHERE a
      = @value` is.
    namedBindings: For each non-reserved named binding site in the query
      string, there must be a named parameter with that name, but not
      necessarily the inverse. Key must match regex `A-Za-z_$*`, must not
      match regex `__.*__`, and must not be `""`.
    positionalBindings: Numbered binding site @1 references the first numbered
      parameter, effectively using 1-based indexing, rather than the usual 0.
      For each binding site numbered i in `query_string`, there must be an
      i-th numbered parameter. The inverse must also be true.
    queryString: A string of the format described
      [here](https://cloud.google.com/datastore/docs/apis/gql/gql_reference).
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class NamedBindingsValue(_messages.Message):
        """For each non-reserved named binding site in the query string, there
    must be a named parameter with that name, but not necessarily the inverse.
    Key must match regex `A-Za-z_$*`, must not match regex `__.*__`, and must
    not be `""`.

    Messages:
      AdditionalProperty: An additional property for a NamedBindingsValue
        object.

    Fields:
      additionalProperties: Additional properties of type NamedBindingsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a NamedBindingsValue object.

      Fields:
        key: Name of the additional property.
        value: A GqlQueryParameter attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GqlQueryParameter', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    allowLiterals = _messages.BooleanField(1)
    namedBindings = _messages.MessageField('NamedBindingsValue', 2)
    positionalBindings = _messages.MessageField('GqlQueryParameter', 3, repeated=True)
    queryString = _messages.StringField(4)