import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _end_title(self):
    if self.svgOK:
        return
    value = self.pop_content('title')
    if not value:
        return
    self.title_depth = self.depth