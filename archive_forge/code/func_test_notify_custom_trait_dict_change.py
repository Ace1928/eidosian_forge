import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._testing import (
from traits.trait_dict_object import TraitDict
from traits.trait_types import Dict, Str
def test_notify_custom_trait_dict_change(self):
    instance = ClassWithDict(custom_trait_dict=CustomTraitDict())
    graph = create_graph(create_observer(notify=True))
    handler = mock.Mock()
    call_add_or_remove_notifiers(object=instance.custom_trait_dict, graph=graph, handler=handler)
    instance.custom_trait_dict.update({'1': 1})
    ((event,), _), = handler.call_args_list
    self.assertEqual(event.added, {'1': 1})
    self.assertEqual(event.removed, {})