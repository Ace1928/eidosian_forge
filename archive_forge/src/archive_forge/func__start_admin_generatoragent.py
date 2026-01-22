from ..util import FeedParserDict
def _start_admin_generatoragent(self, attrs_d):
    self.push('generator', 1)
    value = self._get_attribute(attrs_d, 'rdf:resource')
    if value:
        self.elementstack[-1][2].append(value)
    self.pop('generator')
    self._get_context()['generator_detail'] = FeedParserDict({'href': value})