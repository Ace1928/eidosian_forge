from ..helpers import nativestr
class BFInfo(object):
    capacity = None
    size = None
    filterNum = None
    insertedNum = None
    expansionRate = None

    def __init__(self, args):
        response = dict(zip(map(nativestr, args[::2]), args[1::2]))
        self.capacity = response['Capacity']
        self.size = response['Size']
        self.filterNum = response['Number of filters']
        self.insertedNum = response['Number of items inserted']
        self.expansionRate = response['Expansion rate']

    def get(self, item):
        try:
            return self.__getitem__(item)
        except AttributeError:
            return None

    def __getitem__(self, item):
        return getattr(self, item)