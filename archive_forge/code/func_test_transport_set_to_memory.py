import os
from breezy import tests
from breezy.tests import features
from breezy.transport import memory
def test_transport_set_to_memory(self):
    params = self.get_params_passed_to_core('selftest --transport=memory')
    self.assertEqual(memory.MemoryServer, params[1]['transport'])