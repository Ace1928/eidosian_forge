import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_source(self, attrs_d):
    if 'url' in attrs_d:
        self.sourcedata['href'] = attrs_d['url']
    self.push('source', 1)
    self.insource = 1
    self.title_depth = -1