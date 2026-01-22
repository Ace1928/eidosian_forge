from nltk.internals import Counter
from nltk.sem.logic import APP, LogicParser
class VariableExpression(AtomicExpression):

    def unify(self, other, bindings):
        """
        'self' must not be bound to anything other than 'other'.

        :param other: ``Expression``
        :param bindings: ``BindingDict`` A dictionary of all current bindings
        :return: ``BindingDict`` A new combined dictionary of of 'bindings' and the new binding
        :raise UnificationException: If 'self' and 'other' cannot be unified in the context of 'bindings'
        """
        assert isinstance(other, Expression)
        try:
            if self == other:
                return bindings
            else:
                return bindings + BindingDict([(self, other)])
        except VariableBindingException as e:
            raise UnificationException(self, other, bindings) from e