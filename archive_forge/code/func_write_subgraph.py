import collections
import hashlib
from cliff.formatters import base
def write_subgraph(self, resources, nested_resource):
    stdout = self.stdout
    stack_dot_id = nested_resource.nested_dot_id
    nested_name = nested_resource.resource.resource_name
    stdout.write('  subgraph cluster_%s {\n' % stack_dot_id)
    stdout.write('    label="%s";\n' % nested_name)
    self.write_nodes(resources, 4)
    stdout.write('  }\n\n')