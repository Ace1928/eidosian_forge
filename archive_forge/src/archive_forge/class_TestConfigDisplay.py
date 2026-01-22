from ... import config, tests
from .. import script
from .. import test_config as _t_config
class TestConfigDisplay(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        _t_config.create_configs(self)

    def test_multiline_all_values(self):
        self.breezy_config.set_user_option('multiline', '1\n2\n')
        script.run_script(self, '            $ brz config -d tree\n            breezy:\n              [DEFAULT]\n              multiline = """1\n            2\n            """\n            ')

    def test_multiline_value_only(self):
        self.breezy_config.set_user_option('multiline', '1\n2\n')
        script.run_script(self, '            $ brz config -d tree multiline\n            """1\n            2\n            """\n            ')

    def test_list_value_all(self):
        config.option_registry.register(config.ListOption('list'))
        self.addCleanup(config.option_registry.remove, 'list')
        self.breezy_config.set_user_option('list', [1, 'a', 'with, a comma'])
        script.run_script(self, '            $ brz config -d tree\n            breezy:\n              [DEFAULT]\n              list = 1, a, "with, a comma"\n            ')

    def test_list_value_one(self):
        config.option_registry.register(config.ListOption('list'))
        self.addCleanup(config.option_registry.remove, 'list')
        self.breezy_config.set_user_option('list', [1, 'a', 'with, a comma'])
        script.run_script(self, '            $ brz config -d tree list\n            1, a, "with, a comma"\n            ')

    def test_registry_value_all(self):
        self.breezy_config.set_user_option('transform.orphan_policy', 'move')
        script.run_script(self, '            $ brz config -d tree\n            breezy:\n              [DEFAULT]\n              transform.orphan_policy = move\n            ')

    def test_registry_value_one(self):
        self.breezy_config.set_user_option('transform.orphan_policy', 'move')
        script.run_script(self, '            $ brz config -d tree transform.orphan_policy\n            move\n            ')

    def test_breezy_config(self):
        self.breezy_config.set_user_option('hello', 'world')
        script.run_script(self, '            $ brz config -d tree\n            breezy:\n              [DEFAULT]\n              hello = world\n            ')

    def test_locations_config_for_branch(self):
        self.locations_config.set_user_option('hello', 'world')
        self.branch_config.set_user_option('hello', 'you')
        script.run_script(self, '            $ brz config -d tree\n            locations:\n              [.../tree]\n              hello = world\n            branch:\n              hello = you\n            ')

    def test_locations_config_outside_branch(self):
        self.breezy_config.set_user_option('hello', 'world')
        self.locations_config.set_user_option('hello', 'world')
        script.run_script(self, '            $ brz config\n            breezy:\n              [DEFAULT]\n              hello = world\n            ')

    def test_cmd_line(self):
        self.breezy_config.set_user_option('hello', 'world')
        script.run_script(self, '            $ brz config -Ohello=bzr\n            cmdline:\n              hello = bzr\n            breezy:\n              [DEFAULT]\n              hello = world\n            ')