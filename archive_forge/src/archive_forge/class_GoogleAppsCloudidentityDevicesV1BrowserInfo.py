from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsCloudidentityDevicesV1BrowserInfo(_messages.Message):
    """Browser-specific fields reported by the [Endpoint Verification
  extension](https://chromewebstore.google.com/detail/endpoint-
  verification/callobklhcbilhphinckomhgkigmfocg?pli=1). LINT.IfChange

  Enums:
    BrowserManagementStateValueValuesEnum: Output only. Browser's management
      state.
    PasswordProtectionWarningTriggerValueValuesEnum: Current state of
      [password protection trigger](https://chromeenterprise.google/policies/#
      PasswordProtectionWarningTrigger).
    SafeBrowsingProtectionLevelValueValuesEnum: Current state of [Safe
      Browsing protection level](https://chromeenterprise.google/policies/#Saf
      eBrowsingProtectionLevel).

  Fields:
    browserManagementState: Output only. Browser's management state.
    browserVersion: Version of the request initiating browser.
    isBuiltInDnsClientEnabled: Current state of [built-in DNS client](https://
      chromeenterprise.google/policies/#BuiltInDnsClientEnabled).
    isBulkDataEntryAnalysisEnabled: Current state of [bulk data analysis](http
      s://chromeenterprise.google/policies/#OnBulkDataEntryEnterpriseConnector
      ). Set to true if provider list from Chrome is non-empty.
    isChromeCleanupEnabled: Current state of [Chrome
      Cleanup](https://chromeenterprise.google/policies/#ChromeCleanupEnabled)
      .
    isChromeRemoteDesktopAppBlocked: Current state of [Chrome Remote Desktop
      app](https://chromeenterprise.google/policies/#URLBlocklist).
    isFileDownloadAnalysisEnabled: Current state of [file download analysis](h
      ttps://chromeenterprise.google/policies/#OnFileDownloadedEnterpriseConne
      ctor). Set to true if provider list from Chrome is non-empty.
    isFileUploadAnalysisEnabled: Current state of [file upload analysis](https
      ://chromeenterprise.google/policies/#OnFileAttachedEnterpriseConnector).
      Set to true if provider list from Chrome is non-empty.
    isRealtimeUrlCheckEnabled: Current state of [real-time URL check](https://
      chromeenterprise.google/policies/#EnterpriseRealTimeUrlCheckMode). Set
      to true if provider list from Chrome is non-empty.
    isSecurityEventAnalysisEnabled: Current state of [security event analysis]
      (https://chromeenterprise.google/policies/#OnSecurityEventEnterpriseConn
      ector). Set to true if provider list from Chrome is non-empty.
    isSiteIsolationEnabled: Current state of [site isolation](https://chromeen
      terprise.google/policies/?policy=IsolateOrigins).
    isThirdPartyBlockingEnabled: Current state of [third-party blocking](https
      ://chromeenterprise.google/policies/#ThirdPartyBlockingEnabled).
    passwordProtectionWarningTrigger: Current state of [password protection tr
      igger](https://chromeenterprise.google/policies/#PasswordProtectionWarni
      ngTrigger).
    safeBrowsingProtectionLevel: Current state of [Safe Browsing protection le
      vel](https://chromeenterprise.google/policies/#SafeBrowsingProtectionLev
      el).
  """

    class BrowserManagementStateValueValuesEnum(_messages.Enum):
        """Output only. Browser's management state.

    Values:
      UNSPECIFIED: Management state is not specified.
      UNMANAGED: Browser/Profile is not managed by any customer.
      MANAGED_BY_OTHER_DOMAIN: Browser/Profile is managed, but by some other
        customer.
      PROFILE_MANAGED: Profile is managed by customer.
      BROWSER_MANAGED: Browser is managed by customer.
    """
        UNSPECIFIED = 0
        UNMANAGED = 1
        MANAGED_BY_OTHER_DOMAIN = 2
        PROFILE_MANAGED = 3
        BROWSER_MANAGED = 4

    class PasswordProtectionWarningTriggerValueValuesEnum(_messages.Enum):
        """Current state of [password protection trigger](https://chromeenterpris
    e.google/policies/#PasswordProtectionWarningTrigger).

    Values:
      PASSWORD_PROTECTION_TRIGGER_UNSPECIFIED: Password protection is not
        specified.
      PROTECTION_OFF: Password reuse is never detected.
      PASSWORD_REUSE: Warning is shown when the user reuses their protected
        password on a non-allowed site.
      PHISHING_REUSE: Warning is shown when the user reuses their protected
        password on a phishing site.
    """
        PASSWORD_PROTECTION_TRIGGER_UNSPECIFIED = 0
        PROTECTION_OFF = 1
        PASSWORD_REUSE = 2
        PHISHING_REUSE = 3

    class SafeBrowsingProtectionLevelValueValuesEnum(_messages.Enum):
        """Current state of [Safe Browsing protection level](https://chromeenterp
    rise.google/policies/#SafeBrowsingProtectionLevel).

    Values:
      SAFE_BROWSING_LEVEL_UNSPECIFIED: Browser protection level is not
        specified.
      DISABLED: No protection against dangerous websites, downloads, and
        extensions.
      STANDARD: Standard protection against websites, downloads, and
        extensions that are known to be dangerous.
      ENHANCED: Faster, proactive protection against dangerous websites,
        downloads, and extensions.
    """
        SAFE_BROWSING_LEVEL_UNSPECIFIED = 0
        DISABLED = 1
        STANDARD = 2
        ENHANCED = 3
    browserManagementState = _messages.EnumField('BrowserManagementStateValueValuesEnum', 1)
    browserVersion = _messages.StringField(2)
    isBuiltInDnsClientEnabled = _messages.BooleanField(3)
    isBulkDataEntryAnalysisEnabled = _messages.BooleanField(4)
    isChromeCleanupEnabled = _messages.BooleanField(5)
    isChromeRemoteDesktopAppBlocked = _messages.BooleanField(6)
    isFileDownloadAnalysisEnabled = _messages.BooleanField(7)
    isFileUploadAnalysisEnabled = _messages.BooleanField(8)
    isRealtimeUrlCheckEnabled = _messages.BooleanField(9)
    isSecurityEventAnalysisEnabled = _messages.BooleanField(10)
    isSiteIsolationEnabled = _messages.BooleanField(11)
    isThirdPartyBlockingEnabled = _messages.BooleanField(12)
    passwordProtectionWarningTrigger = _messages.EnumField('PasswordProtectionWarningTriggerValueValuesEnum', 13)
    safeBrowsingProtectionLevel = _messages.EnumField('SafeBrowsingProtectionLevelValueValuesEnum', 14)