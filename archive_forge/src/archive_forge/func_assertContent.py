import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def assertContent(self, branch, tree, path, content):
    with branch.lock_read():
        self.assertEqual(tree.get_file_text(path.decode('utf-8')), content)