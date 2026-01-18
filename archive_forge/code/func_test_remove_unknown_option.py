from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_remove_unknown_option(self):
    self.run_bzr_error(['The "file" configuration option does not exist'], ['config', '--remove', 'file'])