import docutils.utils.math.tex2unichar as tex2unichar
class mfenced(math):
    translation = {'\\{': '{', '\\langle': '〈', '\\}': '}', '\\rangle': '〉', '.': ''}

    def __init__(self, par):
        self.openpar = par
        math.__init__(self)

    def xml_start(self):
        open = self.translation.get(self.openpar, self.openpar)
        close = self.translation.get(self.closepar, self.closepar)
        return ['<mfenced open="%s" close="%s">' % (open, close)]