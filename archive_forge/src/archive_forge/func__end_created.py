import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _end_created(self):
    value = self.pop('created')
    self._save('created_parsed', _parse_date(value), overwrite=True)