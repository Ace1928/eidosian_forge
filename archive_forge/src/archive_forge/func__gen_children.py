from string import Template
import sys
def _gen_children(self):
    src = '    def children(self):\n'
    if self.all_entries:
        src += '        nodelist = []\n'
        for child in self.child:
            src += ('        if self.%(child)s is not None:' + ' nodelist.append(("%(child)s", self.%(child)s))\n') % dict(child=child)
        for seq_child in self.seq_child:
            src += '        for i, child in enumerate(self.%(child)s or []):\n            nodelist.append(("%(child)s[%%d]" %% i, child))\n' % dict(child=seq_child)
        src += '        return tuple(nodelist)\n'
    else:
        src += '        return ()\n'
    return src