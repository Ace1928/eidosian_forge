from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamPolicyAnalysis(_messages.Message):
    """An analysis message to group the query and results.

  Fields:
    analysisQuery: The analysis query.
    analysisResults: A list of IamPolicyAnalysisResult that matches the
      analysis query, or empty if no result is found.
    denyAnalysisResults: A list of DenyAnalysisResult, which contains access
      tuples in the analysis_results that are conducted deny policy analysis.
      The deny policy analysis will be conducted on max 1000 access tuples.
      For access tuples don't have deny policy analysis result populated, you
      can issue another query of that access tuple to get deny policy analysis
      result for it. This is only populated when
      IamPolicyAnalysisQuery.Options.include_deny_policy_analysis is true.
    fullyExplored: Represents whether all entries in the analysis_results have
      been fully explored to answer the query.
    nonCriticalErrors: A list of non-critical errors happened during the query
      handling.
  """
    analysisQuery = _messages.MessageField('IamPolicyAnalysisQuery', 1)
    analysisResults = _messages.MessageField('IamPolicyAnalysisResult', 2, repeated=True)
    denyAnalysisResults = _messages.MessageField('DenyAnalysisResult', 3, repeated=True)
    fullyExplored = _messages.BooleanField(4)
    nonCriticalErrors = _messages.MessageField('IamPolicyAnalysisState', 5, repeated=True)