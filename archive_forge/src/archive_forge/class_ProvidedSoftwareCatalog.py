from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ProvidedSoftwareCatalog(_messages.Message):
    """The currently provided software environment on the devices under test.

  Fields:
    androidxOrchestratorVersion: A string representing the current version of
      AndroidX Test Orchestrator that is used in the environment. The package
      is available at
      https://maven.google.com/web/index.html#androidx.test:orchestrator.
    orchestratorVersion: Deprecated: Use AndroidX Test Orchestrator going
      forward. A string representing the current version of Android Test
      Orchestrator that is used in the environment. The package is available
      at https://maven.google.com/web/index.html#com.android.support.test:orch
      estrator.
  """
    androidxOrchestratorVersion = _messages.StringField(1)
    orchestratorVersion = _messages.StringField(2)