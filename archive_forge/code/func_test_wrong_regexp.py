from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_wrong_regexp(self):
    self.run_bzr_error(['Invalid pattern\\(s\\) found. "\\*file" nothing to repeat'], ['config', '--all', '*file'])