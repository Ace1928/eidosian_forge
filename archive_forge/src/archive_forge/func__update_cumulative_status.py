import contextlib
from datetime import datetime
import sys
import time
def _update_cumulative_status(self):
    """Write an update summarizing the data uploaded since the start."""
    if not self._verbosity:
        return
    if not self._stats.has_new_data_since_last_summarize():
        return
    uploaded_str, skipped_str = self._stats.summarize()
    uploaded_message = '%s[%s]%s Total uploaded: %s\n' % (_STYLE_BOLD, readable_time_string(), _STYLE_RESET, uploaded_str)
    sys.stdout.write(uploaded_message)
    if skipped_str:
        sys.stdout.write('%sTotal skipped: %s\n%s' % (_STYLE_DARKGRAY, skipped_str, _STYLE_RESET))
    sys.stdout.flush()