import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
class TestRegressionNestedContainerEvent(unittest.TestCase):
    """ Regression tests for enthought/traits#281 and enthought/traits#25
    """

    def setUp(self):
        self.events = []

        def change_handler(*args):
            self.events.append(args)
        self.change_handler = change_handler

    def test_modify_list_in_dict(self):
        instance = NestedContainerClass(dict_of_list={'name': []})
        try:
            instance.dict_of_list['name'].append('word')
        except Exception:
            self.fail('Mutating a nested list should not fail.')

    def test_modify_list_in_dict_wrapped_in_union(self):
        instance = NestedContainerClass(dict_of_union_none_or_list={'name': []})
        try:
            instance.dict_of_union_none_or_list['name'].append('word')
        except Exception:
            self.fail('Mutating a nested list should not fail.')

    def test_modify_list_in_list_no_events(self):
        instance = NestedContainerClass(list_of_list=[[]])
        instance.on_trait_change(self.change_handler, 'list_of_list_items')
        instance.list_of_list[0].append(1)
        self.assertEqual(len(self.events), 0, 'Expected no events.')

    def test_modify_dict_in_list(self):
        instance = NestedContainerClass(list_of_dict=[{}])
        try:
            instance.list_of_dict[0]['key'] = 1
        except Exception:
            self.fail('Mutating a nested dict should not fail.')

    def test_modify_dict_in_list_with_new_value(self):
        instance = NestedContainerClass(list_of_dict=[{}])
        instance.list_of_dict.append(dict())
        try:
            instance.list_of_dict[-1]['key'] = 1
        except Exception:
            self.fail('Mutating a nested dict should not fail.')

    def test_modify_dict_in_dict_no_events(self):
        instance = NestedContainerClass(dict_of_dict={'1': {'2': 2}})
        instance.on_trait_change(self.change_handler, 'dict_of_dict_items')
        instance.dict_of_dict['1']['3'] = 3
        self.assertEqual(len(self.events), 0, 'Expected no events.')

    def test_modify_dict_in_union_in_dict_no_events(self):
        instance = NestedContainerClass(dict_of_union_none_or_dict={'1': {'2': 2}})
        instance.on_trait_change(self.change_handler, 'dict_of_union_none_or_dict_items')
        instance.dict_of_union_none_or_dict['1']['3'] = 3
        self.assertEqual(len(self.events), 0, 'Expected no events.')

    def test_modify_set_in_list(self):
        instance = NestedContainerClass(list_of_set=[set()])
        try:
            instance.list_of_set[0].add(1)
        except Exception:
            self.fail('Mutating a nested set should not fail.')

    def test_modify_set_in_list_with_new_value(self):
        instance = NestedContainerClass(list_of_set=[])
        instance.list_of_set.append(set())
        try:
            instance.list_of_set[0].add(1)
        except Exception:
            self.fail('Mutating a nested set should not fail.')

    def test_modify_set_in_dict(self):
        instance = NestedContainerClass(dict_of_set={'1': set()})
        try:
            instance.dict_of_set['1'].add(1)
        except Exception:
            self.fail('Mutating a nested set should not fail.')

    def test_modify_set_in_union_in_dict(self):
        instance = NestedContainerClass(dict_of_union_none_or_set={'1': set()})
        try:
            instance.dict_of_union_none_or_set['1'].add(1)
        except Exception:
            self.fail('Mutating a nested set should not fail.')

    def test_modify_nested_set_no_events(self):
        instance = NestedContainerClass(list_of_set=[set()])
        instance.on_trait_change(self.change_handler, 'list_of_set_items')
        instance.list_of_set[0].add(1)
        self.assertEqual(len(self.events), 0, 'Expected no events.')