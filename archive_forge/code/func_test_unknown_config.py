from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_unknown_config(self):
    self.run_bzr_error(['The "moon" configuration does not exist'], ['config', '--scope', 'moon', '--remove', 'file'])