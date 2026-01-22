from string import Template
import sys
def _gen_init(self):
    src = 'class %s(Node):\n' % self.name
    if self.all_entries:
        args = ', '.join(self.all_entries)
        slots = ', '.join(("'{0}'".format(e) for e in self.all_entries))
        slots += ", 'coord', '__weakref__'"
        arglist = '(self, %s, coord=None)' % args
    else:
        slots = "'coord', '__weakref__'"
        arglist = '(self, coord=None)'
    src += '    __slots__ = (%s)\n' % slots
    src += '    def __init__%s:\n' % arglist
    for name in self.all_entries + ['coord']:
        src += '        self.%s = %s\n' % (name, name)
    return src