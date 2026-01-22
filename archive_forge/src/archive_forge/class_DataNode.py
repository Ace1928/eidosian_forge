from html.parser import HTMLParser
from itertools import zip_longest
class DataNode(Node):
    """
    A Node that contains only string data.
    """

    def __init__(self, data, parent=None):
        super().__init__(parent)
        if not isinstance(data, str):
            raise ValueError('Expecting string type, %s given.' % type(data))
        self._leading_whitespace = ''
        self._trailing_whitespace = ''
        self._stripped_data = ''
        if data == '':
            return
        if data.isspace():
            self._trailing_whitespace = data
            return
        first_non_space = next((idx for idx, ch in enumerate(data) if not ch.isspace()))
        last_non_space = len(data) - next((idx for idx, ch in enumerate(reversed(data)) if not ch.isspace()))
        self._leading_whitespace = data[:first_non_space]
        self._trailing_whitespace = data[last_non_space:]
        self._stripped_data = data[first_non_space:last_non_space]

    @property
    def data(self):
        return f'{self._leading_whitespace}{self._stripped_data}{self._trailing_whitespace}'

    def is_whitespace(self):
        return self._stripped_data == '' and (self._leading_whitespace != '' or self._trailing_whitespace != '')

    def startswith_whitespace(self):
        return self._leading_whitespace != '' or (self._stripped_data == '' and self._trailing_whitespace != '')

    def endswith_whitespace(self):
        return self._trailing_whitespace != '' or (self._stripped_data == '' and self._leading_whitespace != '')

    def lstrip(self):
        if self._leading_whitespace != '':
            self._leading_whitespace = ''
        elif self._stripped_data == '':
            self.rstrip()

    def rstrip(self):
        if self._trailing_whitespace != '':
            self._trailing_whitespace = ''
        elif self._stripped_data == '':
            self.lstrip()

    def collapse_whitespace(self):
        """Noop, ``DataNode.write`` always collapses whitespace"""
        return

    def write(self, doc):
        words = doc.translate_words(self._stripped_data.split())
        str_data = f'{self._leading_whitespace}{' '.join(words)}{self._trailing_whitespace}'
        if str_data != '':
            doc.handle_data(str_data)