from ..util import FeedParserDict
def _start_itunes_category(self, attrs_d):
    self._add_tag(attrs_d.get('text'), 'http://www.itunes.com/', None)
    self.push('category', 1)