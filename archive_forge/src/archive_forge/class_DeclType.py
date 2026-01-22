from inspect import isclass
class DeclType(NamedType):
    """
            Gather the type of a variable
            """

    def generate(self, _):
        return 'typename std::remove_cv<typename std::remove_reference<decltype({0})>::type>::type'.format(self.srepr)