import logging
import os
import re
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.docstringparser import DocStringParser
from botocore.docs.bcdoc.style import ReSTStyle
class ReSTDocument:

    def __init__(self, target='man'):
        self.style = ReSTStyle(self)
        self.target = target
        self.parser = DocStringParser(self)
        self.keep_data = True
        self.do_translation = False
        self.translation_map = {}
        self.hrefs = {}
        self._writes = []
        self._last_doc_string = None

    def _write(self, s):
        if self.keep_data and s is not None:
            self._writes.append(s)

    def write(self, content):
        """
        Write content into the document.
        """
        self._write(content)

    def writeln(self, content):
        """
        Write content on a newline.
        """
        self._write(f'{self.style.spaces()}{content}\n')

    def peek_write(self):
        """
        Returns the last content written to the document without
        removing it from the stack.
        """
        return self._writes[-1]

    def pop_write(self):
        """
        Removes and returns the last content written to the stack.
        """
        return self._writes.pop() if len(self._writes) > 0 else None

    def push_write(self, s):
        """
        Places new content on the stack.
        """
        self._writes.append(s)

    def getvalue(self):
        """
        Returns the current content of the document as a string.
        """
        if self.hrefs:
            self.style.new_paragraph()
            for refname, link in self.hrefs.items():
                self.style.link_target_definition(refname, link)
        return ''.join(self._writes).encode('utf-8')

    def translate_words(self, words):
        return [self.translation_map.get(w, w) for w in words]

    def handle_data(self, data):
        if data and self.keep_data:
            self._write(data)

    def include_doc_string(self, doc_string):
        if doc_string:
            try:
                start = len(self._writes)
                self.parser.feed(doc_string)
                self.parser.close()
                end = len(self._writes)
                self._last_doc_string = (start, end)
            except Exception:
                LOG.debug('Error parsing doc string', exc_info=True)
                LOG.debug(doc_string)

    def remove_last_doc_string(self):
        if self._last_doc_string is not None:
            start, end = self._last_doc_string
            del self._writes[start:end]