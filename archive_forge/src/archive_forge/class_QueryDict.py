from struct import pack, unpack, calcsize
class QueryDict(dict):

    def __getattr__(self, attr):
        try:
            return self.__getitem__(attr)
        except KeyError:
            try:
                return super(QueryDict, self).__getattr__(attr)
            except AttributeError:
                raise KeyError(attr)

    def __setattr__(self, attr, value):
        self.__setitem__(attr, value)