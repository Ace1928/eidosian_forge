from ..util import FeedParserDict
def _end_media_player(self):
    value = self.pop('media_player')
    context = self._get_context()
    context['media_player']['content'] = value