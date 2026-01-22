from nltk.internals import Counter
from nltk.sem.logic import APP, LogicParser
class AtomicExpression(Expression):

    def __init__(self, name, dependencies=None):
        """
        :param name: str for the constant name
        :param dependencies: list of int for the indices on which this atom is dependent
        """
        assert isinstance(name, str)
        self.name = name
        if not dependencies:
            dependencies = []
        self.dependencies = dependencies

    def simplify(self, bindings=None):
        """
        If 'self' is bound by 'bindings', return the atomic to which it is bound.
        Otherwise, return self.

        :param bindings: ``BindingDict`` A dictionary of bindings used to simplify
        :return: ``AtomicExpression``
        """
        if bindings and self in bindings:
            return bindings[self]
        else:
            return self

    def compile_pos(self, index_counter, glueFormulaFactory):
        """
        From Iddo Lev's PhD Dissertation p108-109

        :param index_counter: ``Counter`` for unique indices
        :param glueFormulaFactory: ``GlueFormula`` for creating new glue formulas
        :return: (``Expression``,set) for the compiled linear logic and any newly created glue formulas
        """
        self.dependencies = []
        return (self, [])

    def compile_neg(self, index_counter, glueFormulaFactory):
        """
        From Iddo Lev's PhD Dissertation p108-109

        :param index_counter: ``Counter`` for unique indices
        :param glueFormulaFactory: ``GlueFormula`` for creating new glue formulas
        :return: (``Expression``,set) for the compiled linear logic and any newly created glue formulas
        """
        self.dependencies = []
        return (self, [])

    def initialize_labels(self, fstruct):
        self.name = fstruct.initialize_label(self.name.lower())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.name == other.name

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        accum = self.name
        if self.dependencies:
            accum += '%s' % self.dependencies
        return accum

    def __hash__(self):
        return hash(self.name)