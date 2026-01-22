import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _end_description(self):
    if self._summaryKey == 'content':
        self._end_content()
    else:
        self.pop_content('description')
    self._summaryKey = None