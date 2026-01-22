from inspect import isclass
class Lazy(DependentType):
    """
            A type which can be a reference

            It is used to make a lazy evaluation of numpy expressions

            """

    def generate(self, ctx):
        return 'typename pythonic::lazy<{}>::type'.format(ctx(self.of))