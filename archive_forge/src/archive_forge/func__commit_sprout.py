import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def _commit_sprout(self, tree, name):
    tree.add([name])
    rev_id = tree.commit('rev')
    return (rev_id, tree.controldir.sprout('t2').open_workingtree())