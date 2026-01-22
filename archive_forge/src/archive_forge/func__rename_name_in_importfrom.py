from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import six
from pasta.augment import import_utils
from pasta.base import ast_utils
from pasta.base import scope
def _rename_name_in_importfrom(sc, node, old_name, new_name):
    if old_name == new_name:
        return False
    module_parts = node.module.split('.')
    old_parts = old_name.split('.')
    new_parts = new_name.split('.')
    if module_parts[:len(old_parts)] == old_parts:
        node.module = '.'.join(new_parts + module_parts[len(old_parts):])
        return True
    for alias_to_change in node.names:
        if alias_to_change.name == old_parts[-1]:
            break
    else:
        return False
    alias_to_change.name = new_parts[-1]
    if module_parts != new_parts[:-1]:
        if len(node.names) > 1:
            new_import = import_utils.split_import(sc, node, alias_to_change)
            new_import.module = '.'.join(new_parts[:-1])
        else:
            node.module = '.'.join(new_parts[:-1])
    return True