import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _end_author(self):
    self.pop('author')
    self.inauthor = 0
    self._sync_author_detail()