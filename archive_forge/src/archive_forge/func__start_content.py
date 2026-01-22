import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_content(self, attrs_d):
    self.hasContent = 1
    self.push_content('content', attrs_d, 'text/plain', 1)
    src = attrs_d.get('src')
    if src:
        self.contentparams['src'] = src
    self.push('content', 1)