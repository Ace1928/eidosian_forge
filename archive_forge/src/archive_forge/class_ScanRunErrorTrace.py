from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ScanRunErrorTrace(_messages.Message):
    """Output only. Defines an error trace message for a ScanRun.

  Enums:
    CodeValueValuesEnum: Indicates the error reason code.

  Fields:
    code: Indicates the error reason code.
    mostCommonHttpErrorCode: If the scan encounters TOO_MANY_HTTP_ERRORS, this
      field indicates the most common HTTP error code, if such is available.
      For example, if this code is 404, the scan has encountered too many
      NOT_FOUND responses.
    scanConfigError: If the scan encounters SCAN_CONFIG_ISSUE error, this
      field has the error message encountered during scan configuration
      validation that is performed before each scan run.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """Indicates the error reason code.

    Values:
      CODE_UNSPECIFIED: Default value is never used.
      INTERNAL_ERROR: Indicates that the scan run failed due to an internal
        server error.
      SCAN_CONFIG_ISSUE: Indicates a scan configuration error, usually due to
        outdated ScanConfig settings, such as starting_urls or the DNS
        configuration.
      AUTHENTICATION_CONFIG_ISSUE: Indicates an authentication error, usually
        due to outdated ScanConfig authentication settings.
      TIMED_OUT_WHILE_SCANNING: Indicates a scan operation timeout, usually
        caused by a very large site.
      TOO_MANY_REDIRECTS: Indicates that a scan encountered excessive
        redirects, either to authentication or some other page outside of the
        scan scope.
      TOO_MANY_HTTP_ERRORS: Indicates that a scan encountered numerous errors
        from the web site pages. When available, most_common_http_error_code
        field indicates the most common HTTP error code encountered during the
        scan.
      STARTING_URLS_CRAWL_HTTP_ERRORS: Indicates that some of the starting web
        urls returned HTTP errors during the scan.
    """
        CODE_UNSPECIFIED = 0
        INTERNAL_ERROR = 1
        SCAN_CONFIG_ISSUE = 2
        AUTHENTICATION_CONFIG_ISSUE = 3
        TIMED_OUT_WHILE_SCANNING = 4
        TOO_MANY_REDIRECTS = 5
        TOO_MANY_HTTP_ERRORS = 6
        STARTING_URLS_CRAWL_HTTP_ERRORS = 7
    code = _messages.EnumField('CodeValueValuesEnum', 1)
    mostCommonHttpErrorCode = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    scanConfigError = _messages.MessageField('ScanConfigError', 3)