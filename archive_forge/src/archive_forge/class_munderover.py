import docutils.utils.math.tex2unichar as tex2unichar
class munderover(math):
    nchildren = 3

    def __init__(self, children=None):
        math.__init__(self, children)