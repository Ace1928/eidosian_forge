import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from . import (
class TestBuiltInRegistry(object):
    """Test the built-in registry with the default builders registered."""

    def test_combination(self):
        assert registry.lookup('strict', 'html') == HTMLParserTreeBuilder
        if LXML_PRESENT:
            assert registry.lookup('fast', 'html') == LXMLTreeBuilder
            assert registry.lookup('permissive', 'xml') == LXMLTreeBuilderForXML
        if HTML5LIB_PRESENT:
            assert registry.lookup('html5lib', 'html') == HTML5TreeBuilder

    def test_lookup_by_markup_type(self):
        if LXML_PRESENT:
            assert registry.lookup('html') == LXMLTreeBuilder
            assert registry.lookup('xml') == LXMLTreeBuilderForXML
        else:
            assert registry.lookup('xml') == None
            if HTML5LIB_PRESENT:
                assert registry.lookup('html') == HTML5TreeBuilder
            else:
                assert registry.lookup('html') == HTMLParserTreeBuilder

    def test_named_library(self):
        if LXML_PRESENT:
            assert registry.lookup('lxml', 'xml') == LXMLTreeBuilderForXML
            assert registry.lookup('lxml', 'html') == LXMLTreeBuilder
        if HTML5LIB_PRESENT:
            assert registry.lookup('html5lib') == HTML5TreeBuilder
        assert registry.lookup('html.parser') == HTMLParserTreeBuilder

    def test_beautifulsoup_constructor_does_lookup(self):
        with warnings.catch_warnings(record=True) as w:
            BeautifulSoup('', features='html')
            BeautifulSoup('', features=['html', 'fast'])
            pass
        with pytest.raises(ValueError):
            BeautifulSoup('', features='no-such-feature')