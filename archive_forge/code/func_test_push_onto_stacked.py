from ...builtins import cmd_push
from .. import transport_util, ui_testing
def test_push_onto_stacked(self):
    self.make_branch_and_tree('base', format='1.9')
    self.make_branch_and_tree('source', format='1.9')
    self.start_logging_connections()
    cmd = cmd_push()
    cmd.outf = ui_testing.StringIOWithEncoding()
    cmd.run(self.get_url('remote'), directory='source', stacked_on=self.get_url('base'))
    self.assertEqual(1, len(self.connections))