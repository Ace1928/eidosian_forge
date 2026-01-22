from inspect import isclass
class DependentType(Type):
    """
            A class to be sub-classed by any type that depends on another type
            """

    def __init__(self, of):
        assert of is not None
        super(DependentType, self).__init__(of=of)

    def iscombined(self):
        return self.of.iscombined()