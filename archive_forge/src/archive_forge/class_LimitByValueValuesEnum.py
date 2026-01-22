from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LimitByValueValuesEnum(_messages.Enum):
    """Limit type to use for enforcing this quota limit. Each unique value
    gets the defined number of tokens to consume from. For a quota limit that
    uses user type, each user making requests through the same client
    application project will get his/her own pool of tokens to consume,
    whereas for a limit that uses client project type, all users making
    requests through the same client application project share a single pool
    of tokens.

    Values:
      CLIENT_PROJECT: ID of the project owned by the client application
        developer making the request.
      USER: ID of the end user making the request using the client
        application.
    """
    CLIENT_PROJECT = 0
    USER = 1