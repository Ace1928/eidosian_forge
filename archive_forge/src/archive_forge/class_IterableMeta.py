class IterableMeta(type):

    def __getitem__(cls, item):
        return Iterable(item)