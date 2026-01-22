from __future__ import generator_stop
from ..util import FeedParserDict
def _start_where(self, attrs_d):
    self.push('where', 0)
    context = self._get_context()
    context['where'] = FeedParserDict()