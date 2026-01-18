import collections
import hashlib
from cliff.formatters import base
def write_nested_stack_edges(self):
    stdout = self.stdout
    for dot_id, rinfo in self.resources_by_dot_id.items():
        if rinfo.nested_dot_id:
            nested_resources = self.resources_by_stack[rinfo.nested_dot_id]
            if nested_resources:
                first_resource = list(nested_resources.values())[0]
                stdout.write('  %s -> %s [\n    color=dimgray lhead=cluster_%s arrowhead=none\n  ];\n' % (dot_id, first_resource.res_dot_id, rinfo.nested_dot_id))
    stdout.write('\n')