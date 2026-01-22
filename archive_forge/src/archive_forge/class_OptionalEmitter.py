from pyparsing import (Literal, oneOf, printables, ParserElement, Combine,
class OptionalEmitter(object):

    def __init__(self, expr):
        self.expr = expr

    def makeGenerator(self):

        def optionalGen():
            yield ''
            for s in self.expr.makeGenerator()():
                yield s
        return optionalGen