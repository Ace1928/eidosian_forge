import os
from .... import tests
from ... import upload
from .. import cmds
class AutoPushWithLocation(AutoPushHookTests):

    def setUp(self):
        super().setUp()
        self.make_start_branch()
        conf = self.wt.branch.get_config_stack()
        conf.set('upload_auto', True)
        conf.set('upload_location', self.get_url('target'))
        conf.set('upload_auto_quiet', True)

    def test_auto_push_on_commit(self):
        self.assertPathDoesNotExist('target')
        self.build_tree(['b'])
        self.wt.add(['b'])
        self.wt.commit('two')
        self.assertPathExists('target')
        self.assertPathExists(os.path.join('target', 'a'))
        self.assertPathExists(os.path.join('target', 'b'))

    def test_disable_auto_push(self):
        self.assertPathDoesNotExist('target')
        self.build_tree(['b'])
        self.wt.add(['b'])
        self.wt.commit('two')
        self.wt.branch.get_config_stack().set('upload_auto', False)
        self.build_tree(['c'])
        self.wt.add(['c'])
        self.wt.commit('three')
        self.assertPathDoesNotExist(os.path.join('target', 'c'))