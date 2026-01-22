import re
import html
from paste.util import PySourceColor
def _str2html(src, strip=False, indent_subsequent=0, highlight_inner=False):
    if strip:
        src = src.strip()
    orig_src = src
    try:
        src = PySourceColor.str2html(src, form='snip')
        src = error_re.sub('', src)
        src = pre_re.sub('', src)
        src = re.sub('^[\\n\\r]{0,1}', '', src)
        src = re.sub('[\\n\\r]{0,1}$', '', src)
    except:
        src = html_quote(orig_src)
    lines = src.splitlines()
    if len(lines) == 1:
        return lines[0]
    indent = ' ' * indent_subsequent
    for i in range(1, len(lines)):
        lines[i] = indent + lines[i]
        if highlight_inner and i == len(lines) / 2:
            lines[i] = '<span class="source-highlight">%s</span>' % lines[i]
    src = '<br>\n'.join(lines)
    src = whitespace_re.sub(lambda m: '&nbsp;' * (len(m.group(0)) - 1) + ' ', src)
    return src