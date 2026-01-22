from inspect import isclass
class IndexableContainerType(Type):
    """
            Type of any container of stuff of the same type,
            indexable by another type
            """

    def __init__(self, of_key, of_val):
        super(IndexableContainerType, self).__init__(of_key=of_key, of_val=of_val)

    def iscombined(self):
        return any((of.iscombined() for of in (self.of_key, self.of_val)))

    def generate(self, ctx):
        return 'indexable_container<{0}, typename std::remove_reference<{1}>::type>'.format(ctx(self.of_key), ctx(self.of_val))