import copy
from ..datetimes import _parse_date
from ..urls import make_safe_absolute_uri
from ..util import FeedParserDict
def _start_link(self, attrs_d):
    attrs_d.setdefault('rel', 'alternate')
    if attrs_d['rel'] == 'self':
        attrs_d.setdefault('type', 'application/atom+xml')
    else:
        attrs_d.setdefault('type', 'text/html')
    context = self._get_context()
    attrs_d = self._enforce_href(attrs_d)
    if 'href' in attrs_d:
        attrs_d['href'] = self.resolve_uri(attrs_d['href'])
    expecting_text = self.infeed or self.inentry or self.insource
    context.setdefault('links', [])
    if not (self.inentry and self.inimage):
        context['links'].append(FeedParserDict(attrs_d))
    if 'href' in attrs_d:
        if attrs_d.get('rel') == 'alternate' and self.map_content_type(attrs_d.get('type')) in self.html_types:
            context['link'] = attrs_d['href']
    else:
        self.push('link', expecting_text)