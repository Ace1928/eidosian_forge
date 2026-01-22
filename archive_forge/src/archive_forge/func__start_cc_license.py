from ..util import FeedParserDict
def _start_cc_license(self, attrs_d):
    context = self._get_context()
    value = self._get_attribute(attrs_d, 'rdf:resource')
    attrs_d = FeedParserDict()
    attrs_d['rel'] = 'license'
    if value:
        attrs_d['href'] = value
    context.setdefault('links', []).append(attrs_d)