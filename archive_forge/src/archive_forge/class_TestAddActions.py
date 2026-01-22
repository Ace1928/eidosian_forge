from io import StringIO
from .. import add, errors, tests
from ..bzr import inventory
class TestAddActions(tests.TestCase):

    def test_quiet(self):
        self.run_action('')

    def test__print(self):
        self.run_action('adding path\n')

    def run_action(self, output):
        inv = inventory.Inventory()
        stdout = StringIO()
        action = add.AddAction(to_file=stdout, should_print=bool(output))
        self.apply_redirected(None, stdout, None, action, inv, None, 'path', 'file')
        self.assertEqual(stdout.getvalue(), output)