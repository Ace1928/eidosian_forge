from ..helpers import nativestr
class TopKInfo(object):
    k = None
    width = None
    depth = None
    decay = None

    def __init__(self, args):
        response = dict(zip(map(nativestr, args[::2]), args[1::2]))
        self.k = response['k']
        self.width = response['width']
        self.depth = response['depth']
        self.decay = response['decay']

    def __getitem__(self, item):
        return getattr(self, item)