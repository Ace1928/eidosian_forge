import os
import pickle
import numpy as np
import pytest
from holoviews import (
from holoviews.core.options import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import mpl # noqa
from holoviews.plotting import bokeh # noqa
from holoviews.plotting import plotly # noqa
class TestOptionTreeFind(ComparisonTestCase):

    def setUp(self):
        self.original_option_groups = Options._option_groups[:]
        Options._option_groups = ['group']
        options = OptionTree(groups=['group'])
        self.opts1 = Options('group', kw1='value1')
        self.opts2 = Options('group', kw2='value2')
        self.opts3 = Options('group', kw3='value3')
        self.opts4 = Options('group', kw4='value4')
        self.opts5 = Options('group', kw5='value5')
        self.opts6 = Options('group', kw6='value6')
        options.MyType = self.opts1
        options.XType = self.opts2
        options.MyType.Foo = self.opts3
        options.MyType.Bar = self.opts4
        options.XType.Foo = self.opts5
        options.XType.Bar = self.opts6
        self.options = options
        self.original_options = Store.options()
        Store.options(val=OptionTree(groups=['group']))

    def tearDown(self):
        Options._option_groups = self.original_option_groups
        Store.options(val=self.original_options)
        Store._custom_options = {k: {} for k in Store._custom_options.keys()}

    def test_optiontree_find1(self):
        self.assertEqual(self.options.find('MyType').options('group').options, dict(kw1='value1'))

    def test_optiontree_find2(self):
        self.assertEqual(self.options.find('XType').options('group').options, dict(kw2='value2'))

    def test_optiontree_find3(self):
        self.assertEqual(self.options.find('MyType.Foo').options('group').options, dict(kw1='value1', kw3='value3'))

    def test_optiontree_find4(self):
        self.assertEqual(self.options.find('MyType.Bar').options('group').options, dict(kw1='value1', kw4='value4'))

    def test_optiontree_find5(self):
        self.assertEqual(self.options.find('XType.Foo').options('group').options, dict(kw2='value2', kw5='value5'))

    def test_optiontree_find6(self):
        self.assertEqual(self.options.find('XType.Bar').options('group').options, dict(kw2='value2', kw6='value6'))

    def test_optiontree_find_mismatch1(self):
        self.assertEqual(self.options.find('MyType.Baz').options('group').options, dict(kw1='value1'))

    def test_optiontree_find_mismatch2(self):
        self.assertEqual(self.options.find('XType.Baz').options('group').options, dict(kw2='value2'))

    def test_optiontree_find_mismatch3(self):
        self.assertEqual(self.options.find('Baz').options('group').options, {})

    def test_optiontree_find_mismatch4(self):
        self.assertEqual(self.options.find('Baz.Baz').options('group').options, {})