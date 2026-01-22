import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def assertExecutable(self, branch, tree, path, executable):
    with branch.lock_read():
        self.assertEqual(tree.is_executable(path.decode('utf-8')), executable)