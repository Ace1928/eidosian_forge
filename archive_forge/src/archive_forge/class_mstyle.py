import docutils.utils.math.tex2unichar as tex2unichar
class mstyle(math):

    def __init__(self, children=None, nchildren=None, **kwargs):
        if nchildren is not None:
            self.nchildren = nchildren
        math.__init__(self, children)
        self.attrs = kwargs

    def xml_start(self):
        return ['<mstyle '] + ['%s="%s"' % item for item in list(self.attrs.items())] + ['>']