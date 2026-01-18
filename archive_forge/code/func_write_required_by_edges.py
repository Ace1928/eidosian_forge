import collections
import hashlib
from cliff.formatters import base
def write_required_by_edges(self):
    stdout = self.stdout
    for dot_id, rinfo in self.resources_by_dot_id.items():
        r = rinfo.resource
        required_by = r.required_by
        stack_dot_id = rinfo.stack_dot_id
        if not required_by or not stack_dot_id:
            continue
        stack_resources = self.resources_by_stack.get(stack_dot_id, {})
        for req in required_by:
            other_rinfo = stack_resources.get(req)
            if other_rinfo:
                stdout.write('  %s -> %s;\n' % (rinfo.res_dot_id, other_rinfo.res_dot_id))
    stdout.write('\n')