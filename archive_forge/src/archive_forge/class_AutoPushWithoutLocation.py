import os
from .... import tests
from ... import upload
from .. import cmds
class AutoPushWithoutLocation(AutoPushHookTests):

    def setUp(self):
        super().setUp()
        self.make_start_branch()
        self.wt.branch.get_config_stack().set('upload_auto', True)

    def test_dont_push_if_no_location(self):
        self.assertPathDoesNotExist('target')
        self.build_tree(['b'])
        self.wt.add(['b'])
        self.wt.commit('two')
        self.assertPathDoesNotExist('target')