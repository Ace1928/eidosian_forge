import logging
class BaseStyle:

    def __init__(self, doc, indent_width=2):
        self.doc = doc
        self.indent_width = indent_width
        self._indent = 0
        self.keep_data = True

    @property
    def indentation(self):
        return self._indent

    @indentation.setter
    def indentation(self, value):
        self._indent = value

    def new_paragraph(self):
        return '\n%s' % self.spaces()

    def indent(self):
        self._indent += 1

    def dedent(self):
        if self._indent > 0:
            self._indent -= 1

    def spaces(self):
        return ' ' * (self._indent * self.indent_width)

    def bold(self, s):
        return s

    def ref(self, link, title=None):
        return link

    def h2(self, s):
        return s

    def h3(self, s):
        return s

    def underline(self, s):
        return s

    def italics(self, s):
        return s

    def add_trailing_space_to_previous_write(self):
        last_write = self.doc.pop_write()
        if last_write is None:
            last_write = ''
        if last_write != '' and last_write[-1] != ' ':
            last_write += ' '
        self.doc.push_write(last_write)