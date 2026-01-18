import collections
import hashlib
from cliff.formatters import base
def write_subgraphs(self):
    for dot_id, rinfo in self.resources_by_dot_id.items():
        if rinfo.nested_dot_id:
            resources = self.resources_by_stack[rinfo.nested_dot_id]
            if resources:
                self.write_subgraph(resources, rinfo)