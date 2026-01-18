import collections
import hashlib
from cliff.formatters import base
def write_root_nodes(self):
    for stack_dot_id in set(self.resources_by_stack.keys()).difference(self.nested_stack_ids):
        resources = self.resources_by_stack[stack_dot_id]
        self.write_nodes(resources, 2)