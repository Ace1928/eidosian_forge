from string import Template
import sys
def _gen_iter(self):
    src = '    def __iter__(self):\n'
    if self.all_entries:
        for child in self.child:
            src += ('        if self.%(child)s is not None:\n' + '            yield self.%(child)s\n') % dict(child=child)
        for seq_child in self.seq_child:
            src += '        for child in (self.%(child)s or []):\n            yield child\n' % dict(child=seq_child)
        if not (self.child or self.seq_child):
            src += '        return\n' + '        yield\n'
    else:
        src += '        return\n' + '        yield\n'
    return src