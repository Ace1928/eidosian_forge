from inspect import isclass
class InstantiatedType(Type):
    """
            A type instantiated from a parametric type
            """

    def __init__(self, fun, name, arguments):
        super(InstantiatedType, self).__init__(fun=fun, name=name, arguments=arguments)

    def generate(self, ctx):
        if self.arguments:
            args = ', '.join((ctx(arg) for arg in self.arguments))
            template_params = '<{0}>'.format(args)
        else:
            template_params = ''
        return 'typename {0}::type{1}::{2}'.format(self.fun.name, template_params, self.name)