from ..util import FeedParserDict
def _end_media_credit(self):
    credit = self.pop('credit')
    if credit is not None and credit.strip():
        context = self._get_context()
        context['media_credit'][-1]['content'] = credit