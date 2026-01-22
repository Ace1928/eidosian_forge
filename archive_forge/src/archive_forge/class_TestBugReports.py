import os
import tempfile
import unittest
from pathlib import Path
from bpython.importcompletion import ModuleGatherer
class TestBugReports(unittest.TestCase):

    def test_issue_847(self):
        with tempfile.TemporaryDirectory() as import_test_folder:
            base_path = Path(import_test_folder)
            (base_path / 'xyzzy' / 'plugh').mkdir(parents=True)
            (base_path / 'xyzzy' / '__init__.py').touch()
            (base_path / 'xyzzy' / 'plugh' / '__init__.py').touch()
            (base_path / 'xyzzy' / 'plugh' / 'bar.py').touch()
            (base_path / 'xyzzy' / 'plugh' / 'foo.py').touch()
            module_gatherer = ModuleGatherer((base_path.absolute(),))
            while module_gatherer.find_coroutine():
                pass
            self.assertSetEqual(module_gatherer.complete(17, 'from xyzzy.plugh.'), {'xyzzy.plugh.bar', 'xyzzy.plugh.foo'})