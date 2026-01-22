import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _end_category(self):
    value = self.pop('category')
    if not value:
        return
    context = self._get_context()
    tags = context['tags']
    if value and len(tags) and (not tags[-1]['term']):
        tags[-1]['term'] = value
    else:
        self._add_tag(value, None, None)