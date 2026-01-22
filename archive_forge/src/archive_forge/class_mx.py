import docutils.utils.math.tex2unichar as tex2unichar
class mx(math):
    """Base class for mo, mi, and mn"""
    nchildren = 0

    def __init__(self, data):
        self.data = data

    def xml_body(self):
        return [self.data]