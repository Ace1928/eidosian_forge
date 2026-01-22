from html import escape
from io import StringIO
from incremental import Version
from twisted.python import log
from twisted.python.deprecate import deprecated
@deprecated(Version('Twisted', 15, 3, 0), replacement='twisted.web.template')
def UL(lst):
    io = StringIO()
    io.write('<ul>\n')
    for el in lst:
        io.write('<li> %s</li>\n' % el)
    io.write('</ul>')
    return io.getvalue()