from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListStepAccessibilityClustersResponse(_messages.Message):
    """Response message for AccessibilityService.ListStepAccessibilityClusters.

  Fields:
    clusters: A sequence of accessibility suggestions, grouped into clusters.
      Within the sequence, clusters that belong to the same SuggestionCategory
      should be adjacent. Within each category, clusters should be ordered by
      their SuggestionPriority (ERRORs first). The categories should be
      ordered by their highest priority cluster.
    name: A full resource name of the step. For example, projects/my-
      project/histories/bh.1234567890abcdef/executions/
      1234567890123456789/steps/bs.1234567890abcdef Always presents.
  """
    clusters = _messages.MessageField('SuggestionClusterProto', 1, repeated=True)
    name = _messages.StringField(2)