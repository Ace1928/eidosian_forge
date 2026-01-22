from ..util import FeedParserDict
def _end_itunes_keywords(self):
    for term in self.pop('itunes_keywords').split(','):
        if term.strip():
            self._add_tag(term.strip(), 'http://www.itunes.com/', None)