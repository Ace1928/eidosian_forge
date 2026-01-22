import re
import sys
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
class SectNum(Transform):
    """
    Automatically assigns numbers to the titles of document sections.

    It is possible to limit the maximum section level for which the numbers
    are added.  For those sections that are auto-numbered, the "autonum"
    attribute is set, informing the contents table generator that a different
    form of the TOC should be used.
    """
    default_priority = 710
    'Should be applied before `Contents`.'

    def apply(self):
        self.maxdepth = self.startnode.details.get('depth', None)
        self.startvalue = self.startnode.details.get('start', 1)
        self.prefix = self.startnode.details.get('prefix', '')
        self.suffix = self.startnode.details.get('suffix', '')
        self.startnode.parent.remove(self.startnode)
        if self.document.settings.sectnum_xform:
            if self.maxdepth is None:
                self.maxdepth = sys.maxsize
            self.update_section_numbers(self.document)
        else:
            self.document.settings.sectnum_depth = self.maxdepth
            self.document.settings.sectnum_start = self.startvalue
            self.document.settings.sectnum_prefix = self.prefix
            self.document.settings.sectnum_suffix = self.suffix

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