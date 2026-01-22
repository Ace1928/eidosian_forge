from ..util import FeedParserDict
def _end_itunes_block(self):
    value = self.pop('itunes_block', 0)
    self._get_context()['itunes_block'] = (value == 'yes' or value == 'Yes') and 1 or 0