import copy
import pytest  # NOQA
from .roundtrip import dedent, round_trip_load, round_trip_dump
class TestDeepCopy:

    def test_preserve_flow_style_simple(self):
        x = dedent('        {foo: bar, baz: quux}\n        ')
        data = round_trip_load(x)
        data_copy = copy.deepcopy(data)
        y = round_trip_dump(data_copy)
        print('x [{}]'.format(x))
        print('y [{}]'.format(y))
        assert y == x
        assert data.fa.flow_style() == data_copy.fa.flow_style()

    def test_deepcopy_flow_style_nested_dict(self):
        x = dedent('        a: {foo: bar, baz: quux}\n        ')
        data = round_trip_load(x)
        assert data['a'].fa.flow_style() is True
        data_copy = copy.deepcopy(data)
        assert data_copy['a'].fa.flow_style() is True
        data_copy['a'].fa.set_block_style()
        assert data['a'].fa.flow_style() != data_copy['a'].fa.flow_style()
        assert data['a'].fa._flow_style is True
        assert data_copy['a'].fa._flow_style is False
        y = round_trip_dump(data_copy)
        print('x [{}]'.format(x))
        print('y [{}]'.format(y))
        assert y == dedent('        a:\n          foo: bar\n          baz: quux\n        ')

    def test_deepcopy_flow_style_nested_list(self):
        x = dedent('        a: [1, 2, 3]\n        ')
        data = round_trip_load(x)
        assert data['a'].fa.flow_style() is True
        data_copy = copy.deepcopy(data)
        assert data_copy['a'].fa.flow_style() is True
        data_copy['a'].fa.set_block_style()
        assert data['a'].fa.flow_style() != data_copy['a'].fa.flow_style()
        assert data['a'].fa._flow_style is True
        assert data_copy['a'].fa._flow_style is False
        y = round_trip_dump(data_copy)
        print('x [{}]'.format(x))
        print('y [{}]'.format(y))
        assert y == dedent('        a:\n        - 1\n        - 2\n        - 3\n        ')