import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_textinput(self, attrs_d):
    context = self._get_context()
    context.setdefault('textinput', FeedParserDict())
    self.intextinput = 1
    self.title_depth = -1
    self.push('textinput', 0)