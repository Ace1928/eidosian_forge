from inspect import isclass
class IndexableType(DependentType):
    """
            Type of any container indexed by the same type
            """

    def generate(self, ctx):
        return 'indexable<{0}>'.format(ctx(self.of))