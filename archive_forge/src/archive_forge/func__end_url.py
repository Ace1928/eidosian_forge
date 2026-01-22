import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _end_url(self):
    value = self.pop('href')
    if self.inauthor:
        self._save_author('href', value)
    elif self.incontributor:
        self._save_contributor('href', value)