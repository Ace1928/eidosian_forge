from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
class LogPrinter(object):
    """Formats V2 API log entries to human readable text on a best effort basis.

  A LogPrinter consists of a collection of formatter functions which attempts
  to format specific log entries in a human readable form. The `Format` method
  safely returns a human readable string representation of a log entry, even if
  the provided formatters fails.

  The output format is `{timestamp} {log_text}`, where `timestamp` has a
  configurable but consistent format within a LogPrinter whereas `log_text` is
  emitted from one of its formatters (and truncated if necessary).

  See https://cloud.google.com/logging/docs/api/introduction_v2

  Attributes:
    api_time_format: str, the output format to print. See datetime.strftime()
    max_length: The maximum length of a formatted log entry after truncation.
  """

    def __init__(self, api_time_format='%Y-%m-%d %H:%M:%S', max_length=None):
        self.formatters = []
        self.api_time_format = api_time_format
        self.max_length = max_length

    def Format(self, entry):
        """Safely formats a log entry into human readable text.

    Args:
      entry: A log entry message emitted from the V2 API client.

    Returns:
      A string without line breaks respecting the `max_length` property.
    """
        text = self._LogEntryToText(entry)
        text = text.strip().replace('\n', '  ')
        try:
            time = times.FormatDateTime(times.ParseDateTime(entry.timestamp), self.api_time_format)
        except times.Error:
            log.warning('Received timestamp [{0}] does not match expected format.'.format(entry.timestamp))
            time = '????-??-?? ??:??:??'
        out = '{timestamp} {log_text}'.format(timestamp=time, log_text=text)
        if self.max_length and len(out) > self.max_length:
            out = out[:self.max_length - 3] + '...'
        return out

    def RegisterFormatter(self, formatter):
        """Attach a log entry formatter function to the printer.

    Note that if multiple formatters are attached to the same printer, the first
    added formatter that successfully formats the entry will be used.

    Args:
      formatter: A formatter function which accepts a single argument, a log
          entry. The formatter must either return the formatted log entry as a
          string, or None if it is unable to format the log entry.
          The formatter is allowed to raise exceptions, which will be caught and
          ignored by the printer.
    """
        self.formatters.append(formatter)

    def _LogEntryToText(self, entry):
        """Use the formatters to convert a log entry to unprocessed text."""
        out = None
        for fn in self.formatters + [self._FallbackFormatter]:
            try:
                out = fn(entry)
                if out:
                    break
            except KeyboardInterrupt as e:
                raise e
            except:
                pass
        if not out:
            log.debug('Could not format log entry: %s %s %s', entry.timestamp, entry.logName, entry.insertId)
            out = '< UNREADABLE LOG ENTRY {0}. OPEN THE DEVELOPER CONSOLE TO INSPECT. >'.format(entry.insertId)
        return out

    def _FallbackFormatter(self, entry):
        if entry.protoPayload:
            return six.text_type(entry.protoPayload)
        elif entry.jsonPayload:
            return six.text_type(entry.jsonPayload)
        else:
            return entry.textPayload