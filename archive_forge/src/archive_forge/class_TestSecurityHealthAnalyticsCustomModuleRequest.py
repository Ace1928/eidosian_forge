from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TestSecurityHealthAnalyticsCustomModuleRequest(_messages.Message):
    """Request message to test a SecurityHealthAnalyticsCustomModule.

  Fields:
    securityHealthAnalyticsCustomModule: Custom module to test if provided.
      The name will be ignored in favor of the explicitly specified name
      field.
    testData: Resource data to test against.
  """
    securityHealthAnalyticsCustomModule = _messages.MessageField('GoogleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule', 1)
    testData = _messages.MessageField('TestData', 2, repeated=True)