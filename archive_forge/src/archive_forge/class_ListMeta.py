class ListMeta(type):

    def __getitem__(cls, item):
        return List(item)