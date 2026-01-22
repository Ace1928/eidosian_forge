class SetMeta(type):

    def __getitem__(cls, item):
        return Set(item)