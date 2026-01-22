from io import StringIO
from .. import add, errors, tests
from ..bzr import inventory
def add_helper(self, base_tree, base_path, new_tree, file_list, should_print=False):
    to_file = StringIO()
    base_tree.lock_read()
    try:
        new_tree.lock_write()
        try:
            action = add.AddFromBaseAction(base_tree, base_path, to_file=to_file, should_print=should_print)
            new_tree.smart_add(file_list, action=action)
        finally:
            new_tree.unlock()
    finally:
        base_tree.unlock()
    return to_file.getvalue()