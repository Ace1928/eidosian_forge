class TupleMeta(type):

    def __getitem__(cls, item):
        return Tuple(item)