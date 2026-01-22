from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BehaviorValueValuesEnum(_messages.Enum):
    """Answer this query with a behavior rather than DNS data.

    Values:
      behaviorUnspecified: <no description>
      bypassResponsePolicy: Skip a less-specific ResponsePolicyRule and
        continue normal query logic. This can be used with a less-specific
        wildcard selector to exempt a subset of the wildcard
        ResponsePolicyRule from the ResponsePolicy behavior and query the
        public Internet instead. For instance, if these rules exist:
        *.example.com -> LocalData 1.2.3.4 foo.example.com -> Behavior
        'bypassResponsePolicy' Then a query for 'foo.example.com' skips the
        wildcard. This additionally functions to facilitate the allowlist
        feature. RPZs can be applied to multiple levels in the (eventually
        org, folder, project, network) hierarchy. If a rule is applied at a
        higher level of the hierarchy, adding a passthru rule at a lower level
        will supersede that, and a query from an affected vm to that domain
        will be exempt from the RPZ and proceed to normal resolution behavior.
    """
    behaviorUnspecified = 0
    bypassResponsePolicy = 1