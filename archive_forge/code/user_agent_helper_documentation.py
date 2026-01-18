import six
import sys
import gslib
from gslib.utils import system_util
from gslib.storage_url import StorageUrlFromString
from gslib.exception import InvalidUrlError
Using the command arguments return a suffix for the UserAgent string.

  Args:
    args: str[], parsed set of arguments entered in the CLI.
    metrics_off: boolean, whether the MetricsCollector is disabled.

  Returns:
    str, A string value that can be appended to an existing UserAgent.
  