import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
def _get_active_tags(self):
    global_new, global_gone = self._global_tags
    if self._test_tags is None:
        return set(global_new)
    test_new, test_gone = self._test_tags
    return global_new.difference(test_gone).union(test_new)