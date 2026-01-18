import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_observer_graph_hash_with_named_listener(self):
    path1 = ObserverGraph(node=create_observer(name='foo'), children=[ObserverGraph(node=create_observer(name='bar'))])
    path2 = ObserverGraph(node=create_observer(name='foo'), children=[ObserverGraph(node=create_observer(name='bar'))])
    self.assertEqual(path1, path2)