import re
from ._base import DirectivePlugin
from ..util import escape as escape_text, escape_url
def _parse_attrs(options):
    attrs = {}
    if 'alt' in options:
        attrs['alt'] = options['alt']
    align = options.get('align')
    if align and align in _allowed_aligns:
        attrs['align'] = align
    height = options.get('height')
    width = options.get('width')
    if height and _num_re.match(height):
        attrs['height'] = height
    if width and _num_re.match(width):
        attrs['width'] = width
    if 'target' in options:
        attrs['target'] = escape_url(options['target'])
    return attrs