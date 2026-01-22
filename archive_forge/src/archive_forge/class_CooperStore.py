from nltk.parse import load_parser
from nltk.parse.featurechart import InstantiateVarsChart
from nltk.sem.logic import ApplicationExpression, LambdaExpression, Variable
class CooperStore:
    """
    A container for handling quantifier ambiguity via Cooper storage.
    """

    def __init__(self, featstruct):
        """
        :param featstruct: The value of the ``sem`` node in a tree from
            ``parse_with_bindops()``
        :type featstruct: FeatStruct (with features ``core`` and ``store``)

        """
        self.featstruct = featstruct
        self.readings = []
        try:
            self.core = featstruct['CORE']
            self.store = featstruct['STORE']
        except KeyError:
            print('%s is not a Cooper storage structure' % featstruct)

    def _permute(self, lst):
        """
        :return: An iterator over the permutations of the input list
        :type lst: list
        :rtype: iter
        """
        remove = lambda lst0, index: lst0[:index] + lst0[index + 1:]
        if lst:
            for index, x in enumerate(lst):
                for y in self._permute(remove(lst, index)):
                    yield ((x,) + y)
        else:
            yield ()

    def s_retrieve(self, trace=False):
        """
        Carry out S-Retrieval of binding operators in store. If hack=True,
        serialize the bindop and core as strings and reparse. Ugh.

        Each permutation of the store (i.e. list of binding operators) is
        taken to be a possible scoping of quantifiers. We iterate through the
        binding operators in each permutation, and successively apply them to
        the current term, starting with the core semantic representation,
        working from the inside out.

        Binding operators are of the form::

             bo(\\P.all x.(man(x) -> P(x)),z1)
        """
        for perm, store_perm in enumerate(self._permute(self.store)):
            if trace:
                print('Permutation %s' % (perm + 1))
            term = self.core
            for bindop in store_perm:
                quant, varex = tuple(bindop.args)
                term = ApplicationExpression(quant, LambdaExpression(varex.variable, term))
                if trace:
                    print('  ', term)
                term = term.simplify()
            self.readings.append(term)