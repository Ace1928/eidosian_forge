from inspect import isclass
class LType(Type):

    def __init__(self, base, node):
        super(LType, self).__init__(node=node)
        self.isrec = False
        self.orig = base
        self.final_type = base

    def generate(self, ctx):
        if self.isrec:
            return ctx(self.orig)
        else:
            self.isrec = True
            return ctx(self.final_type)