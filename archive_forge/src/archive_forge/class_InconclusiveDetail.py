from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InconclusiveDetail(_messages.Message):
    """Details for an outcome with an INCONCLUSIVE outcome summary.

  Fields:
    abortedByUser: If the end user aborted the test execution before a pass or
      fail could be determined. For example, the user pressed ctrl-c which
      sent a kill signal to the test runner while the test was running.
    hasErrorLogs: If results are being provided to the user in certain cases
      of infrastructure failures
    infrastructureFailure: If the test runner could not determine success or
      failure because the test depends on a component other than the system
      under test which failed. For example, a mobile test requires
      provisioning a device where the test executes, and that provisioning can
      fail.
  """
    abortedByUser = _messages.BooleanField(1)
    hasErrorLogs = _messages.BooleanField(2)
    infrastructureFailure = _messages.BooleanField(3)