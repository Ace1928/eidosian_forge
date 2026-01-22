from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ErrorGroupStats(_messages.Message):
    """Data extracted for a specific group based on certain filter criteria,
  such as a given time period and/or service filter.

  Fields:
    affectedServices: Service contexts with a non-zero error count for the
      given filter criteria. This list can be truncated if multiple services
      are affected. Refer to `num_affected_services` for the total count.
    affectedUsersCount: Approximate number of affected users in the given
      group that match the filter criteria. Users are distinguished by data in
      the ErrorContext of the individual error events, such as their login
      name or their remote IP address in case of HTTP requests. The number of
      affected users can be zero even if the number of errors is non-zero if
      no data was provided from which the affected user could be deduced.
      Users are counted based on data in the request context that was provided
      in the error report. If more users are implicitly affected, such as due
      to a crash of the whole service, this is not reflected here.
    count: Approximate total number of events in the given group that match
      the filter criteria.
    firstSeenTime: Approximate first occurrence that was ever seen for this
      group and which matches the given filter criteria, ignoring the
      time_range that was specified in the request.
    group: Group data that is independent of the filter criteria.
    lastSeenTime: Approximate last occurrence that was ever seen for this
      group and which matches the given filter criteria, ignoring the
      time_range that was specified in the request.
    numAffectedServices: The total number of services with a non-zero error
      count for the given filter criteria.
    representative: An arbitrary event that is chosen as representative for
      the whole group. The representative event is intended to be used as a
      quick preview for the whole group. Events in the group are usually
      sufficiently similar to each other such that showing an arbitrary
      representative provides insight into the characteristics of the group as
      a whole.
    timedCounts: Approximate number of occurrences over time. Timed counts
      returned by ListGroups are guaranteed to be: - Inside the requested time
      interval - Non-overlapping, and - Ordered by ascending time.
  """
    affectedServices = _messages.MessageField('ServiceContext', 1, repeated=True)
    affectedUsersCount = _messages.IntegerField(2)
    count = _messages.IntegerField(3)
    firstSeenTime = _messages.StringField(4)
    group = _messages.MessageField('ErrorGroup', 5)
    lastSeenTime = _messages.StringField(6)
    numAffectedServices = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    representative = _messages.MessageField('ErrorEvent', 8)
    timedCounts = _messages.MessageField('TimedCount', 9, repeated=True)