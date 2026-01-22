from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppProfile(_messages.Message):
    """A configuration object describing how Cloud Bigtable should treat
  traffic from a particular end user application.

  Enums:
    PriorityValueValuesEnum: This field has been deprecated in favor of
      `standard_isolation.priority`. If you set this field,
      `standard_isolation.priority` will be set instead. The priority of
      requests sent using this app profile.

  Fields:
    dataBoostIsolationReadOnly: Specifies that this app profile is intended
      for read-only usage via the Data Boost feature.
    description: Long form description of the use case for this AppProfile.
    etag: Strongly validated etag for optimistic concurrency control. Preserve
      the value returned from `GetAppProfile` when calling `UpdateAppProfile`
      to fail the request if there has been a modification in the mean time.
      The `update_mask` of the request need not include `etag` for this
      protection to apply. See
      [Wikipedia](https://en.wikipedia.org/wiki/HTTP_ETag) and [RFC
      7232](https://tools.ietf.org/html/rfc7232#section-2.3) for more details.
    multiClusterRoutingUseAny: Use a multi-cluster routing policy.
    name: The unique name of the app profile. Values are of the form
      `projects/{project}/instances/{instance}/appProfiles/_a-zA-Z0-9*`.
    priority: This field has been deprecated in favor of
      `standard_isolation.priority`. If you set this field,
      `standard_isolation.priority` will be set instead. The priority of
      requests sent using this app profile.
    singleClusterRouting: Use a single-cluster routing policy.
    standardIsolation: The standard options used for isolating this app
      profile's traffic from other use cases.
  """

    class PriorityValueValuesEnum(_messages.Enum):
        """This field has been deprecated in favor of
    `standard_isolation.priority`. If you set this field,
    `standard_isolation.priority` will be set instead. The priority of
    requests sent using this app profile.

    Values:
      PRIORITY_UNSPECIFIED: Default value. Mapped to PRIORITY_HIGH (the legacy
        behavior) on creation.
      PRIORITY_LOW: <no description>
      PRIORITY_MEDIUM: <no description>
      PRIORITY_HIGH: <no description>
    """
        PRIORITY_UNSPECIFIED = 0
        PRIORITY_LOW = 1
        PRIORITY_MEDIUM = 2
        PRIORITY_HIGH = 3
    dataBoostIsolationReadOnly = _messages.MessageField('DataBoostIsolationReadOnly', 1)
    description = _messages.StringField(2)
    etag = _messages.StringField(3)
    multiClusterRoutingUseAny = _messages.MessageField('MultiClusterRoutingUseAny', 4)
    name = _messages.StringField(5)
    priority = _messages.EnumField('PriorityValueValuesEnum', 6)
    singleClusterRouting = _messages.MessageField('SingleClusterRouting', 7)
    standardIsolation = _messages.MessageField('StandardIsolation', 8)