import xml.sax
import cgi
from boto.compat import six, StringIO
class ResponseGroup(xml.sax.ContentHandler):
    """A Generic "Response Group", which can
    be anything from the entire list of Items to
    specific response elements within an item"""

    def __init__(self, connection=None, nodename=None):
        """Initialize this Item"""
        self._connection = connection
        self._nodename = nodename
        self._nodepath = []
        self._curobj = None
        self._xml = StringIO()

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.__dict__)

    def get(self, name):
        return self.__dict__.get(name)

    def set(self, name, value):
        self.__dict__[name] = value

    def to_xml(self):
        return '<%s>%s</%s>' % (self._nodename, self._xml.getvalue(), self._nodename)

    def startElement(self, name, attrs, connection):
        self._xml.write('<%s>' % name)
        self._nodepath.append(name)
        if len(self._nodepath) == 1:
            obj = ResponseGroup(self._connection)
            self.set(name, obj)
            self._curobj = obj
        elif self._curobj:
            self._curobj.startElement(name, attrs, connection)
        return None

    def endElement(self, name, value, connection):
        self._xml.write('%s</%s>' % (cgi.escape(value).replace('&amp;amp;', '&amp;'), name))
        if len(self._nodepath) == 0:
            return
        obj = None
        curval = self.get(name)
        if len(self._nodepath) == 1:
            if value or not curval:
                self.set(name, value)
            if self._curobj:
                self._curobj = None
        elif self._curobj:
            self._curobj.endElement(name, value, connection)
        self._nodepath.pop()
        return None