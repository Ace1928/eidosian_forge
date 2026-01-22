from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
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