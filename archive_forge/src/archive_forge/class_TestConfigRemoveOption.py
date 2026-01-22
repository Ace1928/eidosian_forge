from ... import config, tests
from .. import script
from .. import test_config as _t_config
class TestConfigRemoveOption(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        _t_config.create_configs_with_file_option(self)

    def test_unknown_config(self):
        self.run_bzr_error(['The "moon" configuration does not exist'], ['config', '--scope', 'moon', '--remove', 'file'])

    def test_breezy_config_outside_branch(self):
        script.run_script(self, '            $ brz config --scope breezy --remove file\n            $ brz config -d tree --all file\n            locations:\n              [.../work/tree]\n              file = locations\n            branch:\n              file = branch\n            ')

    def test_breezy_config_inside_branch(self):
        script.run_script(self, '            $ brz config -d tree --scope breezy --remove file\n            $ brz config -d tree --all file\n            locations:\n              [.../work/tree]\n              file = locations\n            branch:\n              file = branch\n            ')

    def test_locations_config_inside_branch(self):
        script.run_script(self, '            $ brz config -d tree --scope locations --remove file\n            $ brz config -d tree --all file\n            branch:\n              file = branch\n            breezy:\n              [DEFAULT]\n              file = breezy\n            ')

    def test_branch_config_default(self):
        script.run_script(self, '            $ brz config -d tree --scope locations --remove file\n            $ brz config -d tree --all file\n            branch:\n              file = branch\n            breezy:\n              [DEFAULT]\n              file = breezy\n            ')
        script.run_script(self, '            $ brz config -d tree --remove file\n            $ brz config -d tree --all file\n            breezy:\n              [DEFAULT]\n              file = breezy\n            ')

    def test_branch_config_forcing_branch(self):
        script.run_script(self, '            $ brz config -d tree --scope branch --remove file\n            $ brz config -d tree --all file\n            locations:\n              [.../work/tree]\n              file = locations\n            breezy:\n              [DEFAULT]\n              file = breezy\n            ')
        script.run_script(self, '            $ brz config -d tree --scope locations --remove file\n            $ brz config -d tree --all file\n            breezy:\n              [DEFAULT]\n              file = breezy\n            ')