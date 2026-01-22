from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1TestConfig(_messages.Message):
    """Represents configurations for a test case.

  Fields:
    flow: Flow name to start the test case with. Format:
      `projects//locations//agents//flows/`. Only one of `flow` and `page`
      should be set to indicate the starting point of the test case. If both
      are set, `page` takes precedence over `flow`. If neither is set, the
      test case will start with start page on the default start flow.
    page: The page to start the test case with. Format:
      `projects//locations//agents//flows//pages/`. Only one of `flow` and
      `page` should be set to indicate the starting point of the test case. If
      both are set, `page` takes precedence over `flow`. If neither is set,
      the test case will start with start page on the default start flow.
    trackingParameters: Session parameters to be compared when calculating
      differences.
  """
    flow = _messages.StringField(1)
    page = _messages.StringField(2)
    trackingParameters = _messages.StringField(3, repeated=True)