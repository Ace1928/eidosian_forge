from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DifferentialPrivacyPolicy(_messages.Message):
    """Represents privacy policy associated with "differential privacy" method.

  Fields:
    deltaBudget: Optional. The total delta budget for all queries against the
      privacy-protected view. Each subscriber query against this view charges
      the amount of delta that is pre-defined by the contributor through the
      privacy policy delta_per_query field. If there is sufficient budget,
      then the subscriber query attempts to complete. It might still fail due
      to other reasons, in which case the charge is refunded. If there is
      insufficient budget the query is rejected. There might be multiple
      charge attempts if a single query references multiple views. In this
      case there must be sufficient budget for all charges or the query is
      rejected and charges are refunded in best effort. The budget does not
      have a refresh policy and can only be updated via ALTER VIEW or
      circumvented by creating a new view that can be queried with a fresh
      budget.
    deltaPerQuery: Optional. The delta value that is used per query. Delta
      represents the probability that any row will fail to be epsilon
      differentially private. Indicates the risk associated with exposing
      aggregate rows in the result of a query.
    epsilonBudget: Optional. The total epsilon budget for all queries against
      the privacy-protected view. Each subscriber query against this view
      charges the amount of epsilon they request in their query. If there is
      sufficient budget, then the subscriber query attempts to complete. It
      might still fail due to other reasons, in which case the charge is
      refunded. If there is insufficient budget the query is rejected. There
      might be multiple charge attempts if a single query references multiple
      views. In this case there must be sufficient budget for all charges or
      the query is rejected and charges are refunded in best effort. The
      budget does not have a refresh policy and can only be updated via ALTER
      VIEW or circumvented by creating a new view that can be queried with a
      fresh budget.
    maxEpsilonPerQuery: Optional. The maximum epsilon value that a query can
      consume. If the subscriber specifies epsilon as a parameter in a SELECT
      query, it must be less than or equal to this value. The epsilon
      parameter controls the amount of noise that is added to the groups - a
      higher epsilon means less noise.
    maxGroupsContributed: Optional. The maximum groups contributed value that
      is used per query. Represents the maximum number of groups to which each
      protected entity can contribute. Changing this value does not improve or
      worsen privacy. The best value for accuracy and utility depends on the
      query and data.
    privacyUnitColumn: Optional. The privacy unit column associated with this
      policy. Differential privacy policies can only have one privacy unit
      column per data source object (table, view).
  """
    deltaBudget = _messages.FloatField(1)
    deltaPerQuery = _messages.FloatField(2)
    epsilonBudget = _messages.FloatField(3)
    maxEpsilonPerQuery = _messages.FloatField(4)
    maxGroupsContributed = _messages.IntegerField(5)
    privacyUnitColumn = _messages.StringField(6)