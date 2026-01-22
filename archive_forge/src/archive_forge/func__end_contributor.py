import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _end_contributor(self):
    self.pop('contributor')
    self.incontributor = 0