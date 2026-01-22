from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuotaLimit(_messages.Message):
    """`QuotaLimit` defines a specific limit that applies over a specified
  duration for a limit type. There can be at most one limit for a duration and
  limit type combination defined within a `QuotaGroup`.

  Enums:
    LimitByValueValuesEnum: Limit type to use for enforcing this quota limit.
      Each unique value gets the defined number of tokens to consume from. For
      a quota limit that uses user type, each user making requests through the
      same client application project will get his/her own pool of tokens to
      consume, whereas for a limit that uses client project type, all users
      making requests through the same client application project share a
      single pool of tokens.

  Fields:
    defaultLimit: Default number of tokens that can be consumed during the
      specified duration. This is the number of tokens assigned when a client
      application developer activates the service for his/her project.
      Specifying a value of 0 will block all requests. This can be used if you
      are provisioning quota to selected consumers and blocking others.
      Similarly, a value of -1 will indicate an unlimited quota. No other
      negative values are allowed.
    description: Optional. User-visible, extended description for this quota
      limit. Should be used only when more context is needed to understand
      this limit than provided by the limit's display name (see:
      `display_name`).
    displayName: User-visible display name for this limit. Optional. If not
      set, the UI will provide a default display name based on the quota
      configuration. This field can be used to override the default display
      name generated from the configuration.
    duration: Duration of this limit in textual notation. Example: "100s",
      "24h", "1d". For duration longer than a day, only multiple of days is
      supported. We support only "100s" and "1d" for now. Additional support
      will be added in the future. "0" indicates indefinite duration.
    freeTier: Free tier value displayed in the Developers Console for this
      limit. The free tier is the number of tokens that will be subtracted
      from the billed amount when billing is enabled. This field can only be
      set on a limit with duration "1d", in a billable group; it is invalid on
      any other limit. If this field is not set, it defaults to 0, indicating
      that there is no free tier for this service.
    limitBy: Limit type to use for enforcing this quota limit. Each unique
      value gets the defined number of tokens to consume from. For a quota
      limit that uses user type, each user making requests through the same
      client application project will get his/her own pool of tokens to
      consume, whereas for a limit that uses client project type, all users
      making requests through the same client application project share a
      single pool of tokens.
    maxLimit: Maximum number of tokens that can be consumed during the
      specified duration. Client application developers can override the
      default limit up to this maximum. If specified, this value cannot be set
      to a value less than the default limit. If not specified, it is set to
      the default limit.  To allow clients to apply overrides with no upper
      bound, set this to -1, indicating unlimited maximum quota.
    name: Name of the quota limit.  Must be unique within the quota group.
      This name is used to refer to the limit when overriding the limit on a
      per-project basis.  If a name is not provided, it will be generated from
      the limit_by and duration fields.  The maximum length of the limit name
      is 64 characters.  The name of a limit is used as a unique identifier
      for this limit. Therefore, once a limit has been put into use, its name
      should be immutable. You can use the display_name field to provide a
      user-friendly name for the limit. The display name can be evolved over
      time without affecting the identity of the limit.
  """

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
    defaultLimit = _messages.IntegerField(1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    duration = _messages.StringField(4)
    freeTier = _messages.IntegerField(5)
    limitBy = _messages.EnumField('LimitByValueValuesEnum', 6)
    maxLimit = _messages.IntegerField(7)
    name = _messages.StringField(8)