import collections
import hashlib
from cliff.formatters import base
def write_nodes(self, resources, indent):
    stdout = self.stdout
    spaces = ' ' * indent
    for rinfo in resources.values():
        r = rinfo.resource
        dot_id = rinfo.res_dot_id
        if r.resource_status.endswith('FAILED'):
            style = 'style=filled color=red'
        else:
            style = ''
        stdout.write('%s%s [label="%s\n%s" %s];\n' % (spaces, dot_id, r.resource_name, r.resource_type, style))
    stdout.write('\n')