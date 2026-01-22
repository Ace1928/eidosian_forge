import platform
import re
from io import StringIO
from .. import tests, version, workingtree
from .scenarios import load_tests_apply_scenarios
class TestBzrlibVersioning(tests.TestCase):

    def test_get_brz_source_tree(self):
        """Get tree for bzr source, if any."""
        self.permit_source_tree_branch_repo()
        src_tree = version._get_brz_source_tree()
        if src_tree is None:
            raise tests.TestSkipped("bzr tests aren't run from a bzr working tree")
        else:
            self.assertIsInstance(src_tree, workingtree.WorkingTree)

    def test_python_binary_path(self):
        self.permit_source_tree_branch_repo()
        sio = StringIO()
        version.show_version(show_config=False, show_copyright=False, to_file=sio)
        out = sio.getvalue()
        m = re.search('Python interpreter: (.*) [0-9]', out)
        self.assertIsNot(m, None)
        self.assertPathExists(m.group(1))