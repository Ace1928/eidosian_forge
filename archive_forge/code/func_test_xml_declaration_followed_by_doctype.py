import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.element import SoupStrainer
from . import (
def test_xml_declaration_followed_by_doctype(self):
    markup = '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n<html>\n  <head>\n  </head>\n  <body>\n   <p>foo</p>\n  </body>\n</html>'
    soup = self.soup(markup)
    assert b'<p>foo</p>' == soup.p.encode()