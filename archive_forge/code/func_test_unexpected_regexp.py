from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_unexpected_regexp(self):
    self.run_bzr_error(['The "\\*file" configuration option does not exist'], ['config', '*file'])