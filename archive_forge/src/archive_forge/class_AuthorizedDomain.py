from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthorizedDomain(_messages.Message):
    """A domain that a user has been authorized to administer. To authorize use
  of a domain, verify ownership via [Search
  Console](https://search.google.com/search-console/welcome).

  Fields:
    id: Relative name of the domain authorized for use. Example:
      `example.com`.
    name: Deprecated Read only. Full path to the `AuthorizedDomain` resource
      in the API. Example: `projects/myproject/authorizedDomains/example.com`.
  """
    id = _messages.StringField(1)
    name = _messages.StringField(2)