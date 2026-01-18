import unittest
from unittest.mock import Mock
from IPython.core import events
import IPython.testing.tools as tt
def test_register_unregister(self):
    cb = Mock()
    self.em.register('ping_received', cb)
    self.em.trigger('ping_received')
    self.assertEqual(cb.call_count, 1)
    self.em.unregister('ping_received', cb)
    self.em.trigger('ping_received')
    self.assertEqual(cb.call_count, 1)