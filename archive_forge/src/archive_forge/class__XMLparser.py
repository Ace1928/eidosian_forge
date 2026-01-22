from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
class _XMLparser(ContentHandler):
    """Generic SAX Parser (PRIVATE).

    Just a very basic SAX parser.

    Redefine the methods startElement, characters and endElement.
    """

    def __init__(self, debug=0):
        """Initialize the parser.

        Arguments:
         - debug - integer, amount of debug information to print

        """
        self._tag = []
        self._value = ''
        self._debug = debug
        self._debug_ignore_list = []
        self._method_name_level = 1
        self._method_map = None

    def startElement(self, name, attr):
        """Found XML start tag.

        No real need of attr, BLAST DTD doesn't use them

        Arguments:
         - name -- name of the tag
         - attr -- tag attributes

        """
        self._tag.append(name)
        if len(self._tag) == 1:
            self._on_root_node(name)
            return
        method = 'start_' + self._node_method_name(name)
        if method in self._method_map:
            self._method_map[method]()
            if self._debug > 4:
                print('NCBIXML: Parsed:  ' + method)
        elif self._debug > 3:
            if method not in self._debug_ignore_list:
                print('NCBIXML: Ignored: ' + method)
                self._debug_ignore_list.append(method)
        if self._value.strip():
            raise ValueError(f'What should we do with {self._value} before the {name!r} tag?')
        self._value = ''

    def characters(self, ch):
        """Found some text.

        Arguments:
         - ch -- characters read

        """
        self._value += ch

    def endElement(self, name):
        """Found XML end tag.

        Arguments:
         - name -- tag name

        """
        method = 'end_' + self._node_method_name(name)
        if method in self._method_map:
            self._method_map[method]()
            if self._debug > 2:
                print(f'NCBIXML: Parsed:  {method} {self._value}')
        elif self._debug > 1:
            if method not in self._debug_ignore_list:
                print(f'NCBIXML: Ignored: {method} {self._value}')
                self._debug_ignore_list.append(method)
        self._value = ''
        self._tag.pop()

    def _node_method_name(self, name):
        if self._method_name_level == 1:
            return name
        return '/'.join(self._tag[-self._method_name_level:])