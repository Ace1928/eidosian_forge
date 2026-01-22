import re
from mako import exceptions
def _indent_line(line, stripspace=''):
    return re.sub('^%s' % stripspace, '', line)