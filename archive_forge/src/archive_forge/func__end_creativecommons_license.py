from ..util import FeedParserDict
def _end_creativecommons_license(self):
    value = self.pop('license')
    context = self._get_context()
    attrs_d = FeedParserDict()
    attrs_d['rel'] = 'license'
    if value:
        attrs_d['href'] = value
    context.setdefault('links', []).append(attrs_d)
    del context['license']