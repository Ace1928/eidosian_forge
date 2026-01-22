from .... import msgeditor
from ....tests import TestCaseWithTransport
from ... import commitfromnews
class TestRegisters(TestCaseWithTransport):

    def test_registered_at_import(self):
        self.assertTrue(commitfromnews._registered)

    def test_register_registers_for_commit_message_template(self):
        commitfromnews._registered = False
        commitfromnews.register()
        self.assertLength(1, msgeditor.hooks['commit_message_template'])
        self.assertTrue(commitfromnews._registered)