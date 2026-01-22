import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _cdf_common(self, attrs_d):
    if 'lastmod' in attrs_d:
        self._start_modified({})
        self.elementstack[-1][-1] = attrs_d['lastmod']
        self._end_modified()
    if 'href' in attrs_d:
        self._start_link({})
        self.elementstack[-1][-1] = attrs_d['href']
        self._end_link()