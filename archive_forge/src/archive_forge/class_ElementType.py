from inspect import isclass
class ElementType(Type):
    """
            Type of the ith element of a tuple or container
            """

    def __init__(self, index, of):
        super(ElementType, self).__init__(of=of, index=index)

    def iscombined(self):
        return self.of.iscombined()

    def generate(self, ctx):
        return 'typename std::tuple_element<{0},{1}>::type'.format(self.index, 'typename std::remove_reference<{0}>::type'.format(ctx(self.of)))