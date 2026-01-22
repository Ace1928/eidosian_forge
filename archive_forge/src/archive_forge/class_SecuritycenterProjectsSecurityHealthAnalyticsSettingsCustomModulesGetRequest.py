from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesGetRequest(_messages.Message):
    """A
  SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesGetRequest
  object.

  Fields:
    name: Required. Name of the custom module to get. Its format is "organizat
      ions/{organization}/securityHealthAnalyticsSettings/customModules/{custo
      mModule}", "folders/{folder}/securityHealthAnalyticsSettings/customModul
      es/{customModule}", or "projects/{project}/securityHealthAnalyticsSettin
      gs/customModules/{customModule}"
  """
    name = _messages.StringField(1, required=True)