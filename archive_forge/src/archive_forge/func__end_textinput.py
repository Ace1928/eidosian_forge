import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _end_textinput(self):
    self.pop('textinput')
    self.intextinput = 0