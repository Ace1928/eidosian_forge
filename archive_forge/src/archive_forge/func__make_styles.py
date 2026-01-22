from pygments.formatter import Formatter
from pygments.util import get_bool_opt
def _make_styles(self):
    for ttype, ndef in self.style:
        start = end = ''
        if ndef['color']:
            start += '[color=#%s]' % ndef['color']
            end = '[/color]' + end
        if ndef['bold']:
            start += '[b]'
            end = '[/b]' + end
        if ndef['italic']:
            start += '[i]'
            end = '[/i]' + end
        if ndef['underline']:
            start += '[u]'
            end = '[/u]' + end
        self.styles[ttype] = (start, end)