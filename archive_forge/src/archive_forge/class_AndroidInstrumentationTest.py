from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AndroidInstrumentationTest(_messages.Message):
    """A test of an Android application that can control an Android component
  independently of its normal lifecycle. See for more information on types of
  Android tests.

  Fields:
    testPackageId: The java package for the test to be executed. Required
    testRunnerClass: The InstrumentationTestRunner class. Required
    testTargets: Each target must be fully qualified with the package name or
      class name, in one of these formats: - "package package_name" - "class
      package_name.class_name" - "class package_name.class_name#method_name"
      If empty, all targets in the module will be run.
    useOrchestrator: The flag indicates whether Android Test Orchestrator will
      be used to run test or not.
  """
    testPackageId = _messages.StringField(1)
    testRunnerClass = _messages.StringField(2)
    testTargets = _messages.StringField(3, repeated=True)
    useOrchestrator = _messages.BooleanField(4)