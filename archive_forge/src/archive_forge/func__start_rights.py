import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_rights(self, attrs_d):
    self.push_content('rights', attrs_d, 'text/plain', 1)