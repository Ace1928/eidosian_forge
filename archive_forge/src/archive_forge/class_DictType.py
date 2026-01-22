from inspect import isclass
class DictType(Type):
    """
            Type holding a dict of stuff of the same key and value type
            """

    def __init__(self, of_key, of_val):
        super(DictType, self).__init__(of_key=of_key, of_val=of_val)

    def iscombined(self):
        return any((of.iscombined() for of in (self.of_key, self.of_val)))

    def generate(self, ctx):
        return 'pythonic::types::dict<{},{}>'.format(ctx(self.of_key), ctx(self.of_val))