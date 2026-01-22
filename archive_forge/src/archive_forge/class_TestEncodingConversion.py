from pdb import set_trace
import logging
import os
import pickle
import pytest
import sys
import tempfile
from bs4 import (
from bs4.builder import (
from bs4.element import (
from . import (
import warnings
class TestEncodingConversion(SoupTest):

    def setup_method(self):
        self.unicode_data = '<html><head><meta charset="utf-8"/></head><body><foo>Sacré bleu!</foo></body></html>'
        self.utf8_data = self.unicode_data.encode('utf-8')
        assert self.utf8_data == b'<html><head><meta charset="utf-8"/></head><body><foo>Sacr\xc3\xa9 bleu!</foo></body></html>'

    def test_ascii_in_unicode_out(self):
        chardet = dammit.chardet_dammit
        logging.disable(logging.WARNING)
        try:

            def noop(str):
                return None
            dammit.chardet_dammit = noop
            ascii = b'<foo>a</foo>'
            soup_from_ascii = self.soup(ascii)
            unicode_output = soup_from_ascii.decode()
            assert isinstance(unicode_output, str)
            assert unicode_output == self.document_for(ascii.decode())
            assert soup_from_ascii.original_encoding.lower() == 'utf-8'
        finally:
            logging.disable(logging.NOTSET)
            dammit.chardet_dammit = chardet

    def test_unicode_in_unicode_out(self):
        soup_from_unicode = self.soup(self.unicode_data)
        assert soup_from_unicode.decode() == self.unicode_data
        assert soup_from_unicode.foo.string == 'Sacré bleu!'
        assert soup_from_unicode.original_encoding == None

    def test_utf8_in_unicode_out(self):
        soup_from_utf8 = self.soup(self.utf8_data)
        assert soup_from_utf8.decode() == self.unicode_data
        assert soup_from_utf8.foo.string == 'Sacré bleu!'

    def test_utf8_out(self):
        soup_from_unicode = self.soup(self.unicode_data)
        assert soup_from_unicode.encode('utf-8') == self.utf8_data