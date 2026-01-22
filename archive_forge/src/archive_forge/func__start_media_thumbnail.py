from ..util import FeedParserDict
def _start_media_thumbnail(self, attrs_d):
    context = self._get_context()
    context.setdefault('media_thumbnail', [])
    self.push('url', 1)
    context['media_thumbnail'].append(attrs_d)