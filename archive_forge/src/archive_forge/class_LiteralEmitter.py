from pyparsing import (Literal, oneOf, printables, ParserElement, Combine,
class LiteralEmitter(object):

    def __init__(self, lit):
        self.lit = lit

    def __str__(self):
        return 'Lit:' + self.lit

    def __repr__(self):
        return 'Lit:' + self.lit

    def makeGenerator(self):

        def litGen():
            yield self.lit
        return litGen