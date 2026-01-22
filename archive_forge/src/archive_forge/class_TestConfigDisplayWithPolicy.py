from ... import config, tests
from .. import script
from .. import test_config as _t_config
class TestConfigDisplayWithPolicy(tests.TestCaseWithTransport):

    def test_location_with_policy(self):
        self.make_branch_and_tree('tree')
        config_text = '[{dir}]\nurl = dir\nurl:policy = appendpath\n[{dir}/tree]\nurl = tree\n'.format(dir=self.test_dir)
        config.LocationConfig.from_string(config_text, 'tree', save=True)
        script.run_script(self, '            $ brz config -d tree --all url\n            locations:\n              [.../work/tree]\n              url = tree\n              [.../work]\n              url = dir\n              url:policy = appendpath\n            ')