from __future__ import absolute_import, unicode_literals
import re
from abc import ABCMeta, abstractmethod
from unittest import TestCase
import pytest
import six
from pybtex import textutils
from pybtex.richtext import HRef, Protected, String, Symbol, Tag, Text, nbsp
class TestSymbol(TextTestMixin, TestCase):

    def test__init__(self):
        assert nbsp.name == 'nbsp'

    def test__eq__(self):
        assert Symbol('nbsp') == Symbol('nbsp')
        assert not Symbol('nbsp') != Symbol('nbsp')
        assert not Symbol('nbsp') == Symbol('ndash')
        assert Symbol('nbsp') != Symbol('ndash')
        assert Text(nbsp, nbsp) == Text(Symbol('nbsp'), Symbol('nbsp'))

    def test__str__(self):
        assert six.text_type(nbsp) == '<nbsp>'

    def test__len__(self):
        assert len(nbsp) == 1

    def test__contains__(self):
        assert not '' in nbsp
        assert not 'abc' in nbsp

    def test__getitem__(self):
        symbol = Symbol('nbsp')
        assert symbol[0] == Symbol('nbsp')
        assert symbol[0:] == Symbol('nbsp')
        assert symbol[0:5] == Symbol('nbsp')
        assert symbol[1:] == String()
        assert symbol[1:5] == String()
        with pytest.raises(IndexError):
            symbol[1]

    def test__add__(self):
        assert (nbsp + '.').render_as('html') == '&nbsp;.'

    def test_split(self):
        assert nbsp.split() == [nbsp]
        text = Text('F.', nbsp, 'Miller')
        assert text.split() == [text]

    def test_join(self):
        assert nbsp.join(['S.', 'Jerusalem']) == Text('S.', nbsp, 'Jerusalem')

    def test_upper(self):
        assert nbsp.upper().render_as('html') == '&nbsp;'

    def test_lower(self):
        assert nbsp.lower().render_as('html') == '&nbsp;'

    def test_capfirst(self):
        assert Text(nbsp, nbsp).capfirst().render_as('html') == '&nbsp;&nbsp;'

    def test_capitalize(self):
        assert Text(nbsp, nbsp).capitalize().render_as('html') == '&nbsp;&nbsp;'

    def test_add_period(self):
        assert nbsp.add_period().render_as('html') == '&nbsp;.'
        assert nbsp.add_period().add_period().render_as('html') == '&nbsp;.'

    def test_append(self):
        assert nbsp.append('.').render_as('html') == '&nbsp;.'

    def test_startswith(self):
        assert not nbsp.startswith('.')
        assert not nbsp.startswith(('.', '?!'))

    def test_endswith(self):
        assert not nbsp.endswith('.')
        assert not nbsp.endswith(('.', '?!'))

    def test_isalpha(self):
        assert not nbsp.isalpha()

    def test_render_as(self):
        assert nbsp.render_as('latex') == '~'
        assert nbsp.render_as('html') == '&nbsp;'
        assert Text(nbsp, nbsp).render_as('html') == '&nbsp;&nbsp;'