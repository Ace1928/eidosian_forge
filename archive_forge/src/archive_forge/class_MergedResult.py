from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MergedResult(_messages.Message):
    """Merged test result for environment. If the environment has only one step
  (no reruns or shards), then the merged result is the same as the step
  result. If the environment has multiple shards and/or reruns, then the
  results of shards and reruns that belong to the same environment are merged
  into one environment result.

  Enums:
    StateValueValuesEnum: State of the resource

  Fields:
    outcome: Outcome of the resource
    state: State of the resource
    testSuiteOverviews: The combined and rolled-up result of each test suite
      that was run as part of this environment. Combining: When the test cases
      from a suite are run in different steps (sharding), the results are
      added back together in one overview. (e.g., if shard1 has 2 failures and
      shard2 has 1 failure than the overview failure_count = 3). Rollup: When
      test cases from the same suite are run multiple times (flaky), the
      results are combined (e.g., if testcase1.run1 fails, testcase1.run2
      passes, and both testcase2.run1 and testcase2.run2 fail then the
      overview flaky_count = 1 and failure_count = 1).
  """

    class StateValueValuesEnum(_messages.Enum):
        """State of the resource

    Values:
      unknownState: Should never be in this state. Exists for proto
        deserialization backward compatibility.
      pending: The Execution/Step is created, ready to run, but not running
        yet. If an Execution/Step is created without initial state, it is
        assumed that the Execution/Step is in PENDING state.
      inProgress: The Execution/Step is in progress.
      complete: The finalized, immutable state. Steps/Executions in this state
        cannot be modified.
    """
        unknownState = 0
        pending = 1
        inProgress = 2
        complete = 3
    outcome = _messages.MessageField('Outcome', 1)
    state = _messages.EnumField('StateValueValuesEnum', 2)
    testSuiteOverviews = _messages.MessageField('TestSuiteOverview', 3, repeated=True)