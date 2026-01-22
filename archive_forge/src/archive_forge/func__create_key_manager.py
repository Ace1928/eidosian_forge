from castellan.key_manager import not_implemented_key_manager
from castellan.tests.unit.key_manager import test_key_manager
def _create_key_manager(self):
    return not_implemented_key_manager.NotImplementedKeyManager()