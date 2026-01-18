from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_registry_value_one(self):
    self.breezy_config.set_user_option('transform.orphan_policy', 'move')
    script.run_script(self, '            $ brz config -d tree transform.orphan_policy\n            move\n            ')