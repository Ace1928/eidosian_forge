from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
class TestFindAllByAttribute(SoupTest):

    def test_find_all_by_attribute_name(self):
        tree = self.soup('\n                         <a id="first">Matching a.</a>\n                         <a id="second">\n                          Non-matching <b id="first">Matching b.</b>a.\n                         </a>')
        self.assert_selects(tree.find_all(id='first'), ['Matching a.', 'Matching b.'])

    def test_find_all_by_utf8_attribute_value(self):
        peace = 'םולש'.encode('utf8')
        data = '<a title="םולש"></a>'.encode('utf8')
        soup = self.soup(data)
        assert [soup.a] == soup.find_all(title=peace)
        assert [soup.a] == soup.find_all(title=peace.decode('utf8'))
        assert [soup.a], soup.find_all(title=[peace, 'something else'])

    def test_find_all_by_attribute_dict(self):
        tree = self.soup('\n                         <a name="name1" class="class1">Name match.</a>\n                         <a name="name2" class="class2">Class match.</a>\n                         <a name="name3" class="class3">Non-match.</a>\n                         <name1>A tag called \'name1\'.</name1>\n                         ')
        self.assert_selects(tree.find_all(name='name1'), ["A tag called 'name1'."])
        self.assert_selects(tree.find_all(attrs={'name': 'name1'}), ['Name match.'])
        self.assert_selects(tree.find_all(attrs={'class': 'class2'}), ['Class match.'])

    def test_find_all_by_class(self):
        tree = self.soup('\n                         <a class="1">Class 1.</a>\n                         <a class="2">Class 2.</a>\n                         <b class="1">Class 1.</b>\n                         <c class="3 4">Class 3 and 4.</c>\n                         ')
        self.assert_selects(tree.find_all('a', class_='1'), ['Class 1.'])
        self.assert_selects(tree.find_all('c', class_='3'), ['Class 3 and 4.'])
        self.assert_selects(tree.find_all('c', class_='4'), ['Class 3 and 4.'])
        self.assert_selects(tree.find_all('a', '1'), ['Class 1.'])
        self.assert_selects(tree.find_all(attrs='1'), ['Class 1.', 'Class 1.'])
        self.assert_selects(tree.find_all('c', '3'), ['Class 3 and 4.'])
        self.assert_selects(tree.find_all('c', '4'), ['Class 3 and 4.'])

    def test_find_by_class_when_multiple_classes_present(self):
        tree = self.soup("<gar class='foo bar'>Found it</gar>")
        f = tree.find_all('gar', class_=re.compile('o'))
        self.assert_selects(f, ['Found it'])
        f = tree.find_all('gar', class_=re.compile('a'))
        self.assert_selects(f, ['Found it'])
        f = tree.find_all('gar', class_=re.compile('o b'))
        self.assert_selects(f, ['Found it'])

    def test_find_all_with_non_dictionary_for_attrs_finds_by_class(self):
        soup = self.soup("<a class='bar'>Found it</a>")
        self.assert_selects(soup.find_all('a', re.compile('ba')), ['Found it'])

        def big_attribute_value(value):
            return len(value) > 3
        self.assert_selects(soup.find_all('a', big_attribute_value), [])

        def small_attribute_value(value):
            return len(value) <= 3
        self.assert_selects(soup.find_all('a', small_attribute_value), ['Found it'])

    def test_find_all_with_string_for_attrs_finds_multiple_classes(self):
        soup = self.soup('<a class="foo bar"></a><a class="foo"></a>')
        a, a2 = soup.find_all('a')
        assert [a, a2], soup.find_all('a', 'foo')
        assert [a], soup.find_all('a', 'bar')
        assert [a] == soup.find_all('a', class_='foo bar')
        assert [a] == soup.find_all('a', 'foo bar')
        assert [] == soup.find_all('a', 'bar foo')

    def test_find_all_by_attribute_soupstrainer(self):
        tree = self.soup('\n                         <a id="first">Match.</a>\n                         <a id="second">Non-match.</a>')
        strainer = SoupStrainer(attrs={'id': 'first'})
        self.assert_selects(tree.find_all(strainer), ['Match.'])

    def test_find_all_with_missing_attribute(self):
        tree = self.soup('<a id="1">ID present.</a>\n                            <a>No ID present.</a>\n                            <a id="">ID is empty.</a>')
        self.assert_selects(tree.find_all('a', id=None), ['No ID present.'])

    def test_find_all_with_defined_attribute(self):
        tree = self.soup('<a id="1">ID present.</a>\n                            <a>No ID present.</a>\n                            <a id="">ID is empty.</a>')
        self.assert_selects(tree.find_all(id=True), ['ID present.', 'ID is empty.'])

    def test_find_all_with_numeric_attribute(self):
        tree = self.soup('<a id=1>Unquoted attribute.</a>\n                            <a id="1">Quoted attribute.</a>')
        expected = ['Unquoted attribute.', 'Quoted attribute.']
        self.assert_selects(tree.find_all(id=1), expected)
        self.assert_selects(tree.find_all(id='1'), expected)

    def test_find_all_with_list_attribute_values(self):
        tree = self.soup('<a id="1">1</a>\n                            <a id="2">2</a>\n                            <a id="3">3</a>\n                            <a>No ID.</a>')
        self.assert_selects(tree.find_all(id=['1', '3', '4']), ['1', '3'])

    def test_find_all_with_regular_expression_attribute_value(self):
        tree = self.soup('<a id="a">One a.</a>\n                            <a id="aa">Two as.</a>\n                            <a id="ab">Mixed as and bs.</a>\n                            <a id="b">One b.</a>\n                            <a>No ID.</a>')
        self.assert_selects(tree.find_all(id=re.compile('^a+$')), ['One a.', 'Two as.'])

    def test_find_by_name_and_containing_string(self):
        soup = self.soup('<b>foo</b><b>bar</b><a>foo</a>')
        a = soup.a
        assert [a] == soup.find_all('a', string='foo')
        assert [] == soup.find_all('a', string='bar')

    def test_find_by_name_and_containing_string_when_string_is_buried(self):
        soup = self.soup('<a>foo</a><a><b><c>foo</c></b></a>')
        assert soup.find_all('a') == soup.find_all('a', string='foo')

    def test_find_by_attribute_and_containing_string(self):
        soup = self.soup('<b id="1">foo</b><a id="2">foo</a>')
        a = soup.a
        assert [a] == soup.find_all(id=2, string='foo')
        assert [] == soup.find_all(id=1, string='bar')