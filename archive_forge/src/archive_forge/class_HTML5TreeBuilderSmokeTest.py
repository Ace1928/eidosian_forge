import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
class HTML5TreeBuilderSmokeTest(HTMLTreeBuilderSmokeTest):
    """Smoke test for a tree builder that supports HTML5."""

    def test_real_xhtml_document(self):
        pass

    def test_html_tags_have_namespace(self):
        markup = '<a>'
        soup = self.soup(markup)
        assert 'http://www.w3.org/1999/xhtml' == soup.a.namespace

    def test_svg_tags_have_namespace(self):
        markup = '<svg><circle/></svg>'
        soup = self.soup(markup)
        namespace = 'http://www.w3.org/2000/svg'
        assert namespace == soup.svg.namespace
        assert namespace == soup.circle.namespace

    def test_mathml_tags_have_namespace(self):
        markup = '<math><msqrt>5</msqrt></math>'
        soup = self.soup(markup)
        namespace = 'http://www.w3.org/1998/Math/MathML'
        assert namespace == soup.math.namespace
        assert namespace == soup.msqrt.namespace

    def test_xml_declaration_becomes_comment(self):
        markup = '<?xml version="1.0" encoding="utf-8"?><html></html>'
        soup = self.soup(markup)
        assert isinstance(soup.contents[0], Comment)
        assert soup.contents[0] == '?xml version="1.0" encoding="utf-8"?'
        assert 'html' == soup.contents[0].next_element.name