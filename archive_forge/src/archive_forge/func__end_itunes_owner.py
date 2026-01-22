from ..util import FeedParserDict
def _end_itunes_owner(self):
    self.pop('publisher')
    self.inpublisher = 0
    self._sync_author_detail('publisher')