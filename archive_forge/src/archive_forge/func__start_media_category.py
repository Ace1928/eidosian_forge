from ..util import FeedParserDict
def _start_media_category(self, attrs_d):
    attrs_d.setdefault('scheme', 'http://search.yahoo.com/mrss/category_schema')
    self._start_category(attrs_d)