from testtools.matchers import Equals, Matcher, Mismatch
from .. import osutils
from .. import revision as _mod_revision
from ..tree import InterTree, TreeChange
@staticmethod
def _convert_to_inventory_tree_changes(old_tree, new_tree, expected):
    from ..bzr.inventorytree import InventoryTreeChange
    rich_expected = []

    def get_parent_id(t, p):
        if p:
            return t.path2id(osutils.dirname(p))
        else:
            return None
    for c in expected:
        if c.path[0] is not None:
            file_id = old_tree.path2id(c.path[0])
        else:
            file_id = new_tree.path2id(c.path[1])
        old_parent_id = get_parent_id(old_tree, c.path[0])
        new_parent_id = get_parent_id(new_tree, c.path[1])
        rich_expected.append(InventoryTreeChange(file_id=file_id, parent_id=(old_parent_id, new_parent_id), path=c.path, changed_content=c.changed_content, versioned=c.versioned, name=c.name, kind=c.kind, executable=c.executable, copied=c.copied))
    return rich_expected