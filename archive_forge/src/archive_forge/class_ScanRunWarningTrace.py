from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ScanRunWarningTrace(_messages.Message):
    """Output only. Defines a warning trace message for ScanRun. Warning traces
  provide customers with useful information that helps make the scanning
  process more effective.

  Enums:
    CodeValueValuesEnum: Indicates the warning code.

  Fields:
    code: Indicates the warning code.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """Indicates the warning code.

    Values:
      CODE_UNSPECIFIED: Default value is never used.
      INSUFFICIENT_CRAWL_RESULTS: Indicates that a scan discovered an
        unexpectedly low number of URLs. This is sometimes caused by complex
        navigation features or by using a single URL for numerous pages.
      TOO_MANY_CRAWL_RESULTS: Indicates that a scan discovered too many URLs
        to test, or excessive redundant URLs.
      TOO_MANY_FUZZ_TASKS: Indicates that too many tests have been generated
        for the scan. Customer should try reducing the number of starting
        URLs, increasing the QPS rate, or narrowing down the scope of the scan
        using the excluded patterns.
      BLOCKED_BY_IAP: Indicates that a scan is blocked by IAP.
      NO_STARTING_URL_FOUND_FOR_MANAGED_SCAN: Indicates that no seed is found
        for a scan
    """
        CODE_UNSPECIFIED = 0
        INSUFFICIENT_CRAWL_RESULTS = 1
        TOO_MANY_CRAWL_RESULTS = 2
        TOO_MANY_FUZZ_TASKS = 3
        BLOCKED_BY_IAP = 4
        NO_STARTING_URL_FOUND_FOR_MANAGED_SCAN = 5
    code = _messages.EnumField('CodeValueValuesEnum', 1)