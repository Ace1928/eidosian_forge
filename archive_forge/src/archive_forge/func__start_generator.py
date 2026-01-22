import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_generator(self, attrs_d):
    if attrs_d:
        attrs_d = self._enforce_href(attrs_d)
        if 'href' in attrs_d:
            attrs_d['href'] = self.resolve_uri(attrs_d['href'])
    self._get_context()['generator_detail'] = FeedParserDict(attrs_d)
    self.push('generator', 1)