from ..util import FeedParserDict
def _start_media_restriction(self, attrs_d):
    context = self._get_context()
    context.setdefault('media_restriction', attrs_d)
    self.push('restriction', 1)