from ..util import FeedParserDict
def _end_itunes_explicit(self):
    value = self.pop('itunes_explicit', 0)
    self._get_context()['itunes_explicit'] = (None, False, True)[value == 'yes' and 2 or value == 'clean' or 0]