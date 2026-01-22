from ..pari import pari
import string
from itertools import combinations, combinations_with_replacement, product
class Presentation:
    """
    Class representing a presentation of a finitely presented group.
    gens is a list of Word objects representing the generators
    rels is a list of Word objects representing the relations.
    """

    def __init__(self, G, R):
        """Creates a Presentation from G:generators and R:relations"""
        self.gens = G
        self.rels = R

    def __repr__(self):
        r = ''
        for w in self.rels:
            r += '\n' + str(w)
        return 'Generators\n' + str(self.gens) + '\n' + 'Relations' + r