import re
import sys
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def update_section_numbers(self, node, prefix=(), depth=0):
    depth += 1
    if prefix:
        sectnum = 1
    else:
        sectnum = self.startvalue
    for child in node:
        if isinstance(child, nodes.section):
            numbers = prefix + (str(sectnum),)
            title = child[0]
            generated = nodes.generated('', self.prefix + '.'.join(numbers) + self.suffix + '\xa0' * 3, classes=['sectnum'])
            title.insert(0, generated)
            title['auto'] = 1
            if depth < self.maxdepth:
                self.update_section_numbers(child, numbers, depth)
            sectnum += 1