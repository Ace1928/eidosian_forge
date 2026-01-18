from .... import branch, errors, osutils, tests
from ....bzr import inventory
from .. import revision_store
from . import FastimportFeature
def test_path2id(self):
    basis_inv = self.make_trivial_basis_inv()
    shim = revision_store._TreeShim(repo=None, basis_inv=basis_inv, inv_delta=[], content_provider=None)
    self.assertEqual(b'TREE_ROOT', shim.path2id(''))
    self.assertEqual(b'bar-id', shim.path2id('bar'))