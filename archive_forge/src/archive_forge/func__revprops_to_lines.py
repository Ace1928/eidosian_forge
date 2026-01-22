from copy import copy
from ..osutils import contains_linebreaks, contains_whitespace, sha_strings
from ..tree import Tree
def _revprops_to_lines(self):
    """Pack up revision properties."""
    if not self.revprops:
        return []
    r = ['properties:\n']
    for name, value in sorted(self.revprops.items()):
        if contains_whitespace(name):
            raise ValueError(name)
        r.append('  %s:\n' % name)
        for line in value.splitlines():
            r.append('    %s\n' % line)
    return r