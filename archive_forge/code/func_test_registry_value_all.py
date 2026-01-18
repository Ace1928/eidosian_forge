from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_registry_value_all(self):
    self.breezy_config.set_user_option('transform.orphan_policy', 'move')
    script.run_script(self, '            $ brz config -d tree\n            breezy:\n              [DEFAULT]\n              transform.orphan_policy = move\n            ')