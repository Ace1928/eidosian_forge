import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
class BindingDict:

    def __init__(self, binding_list=None):
        """
        :param binding_list: list of (``AbstractVariableExpression``, ``AtomicExpression``) to initialize the dictionary
        """
        self.d = {}
        if binding_list:
            for v, b in binding_list:
                self[v] = b

    def __setitem__(self, variable, binding):
        """
        A binding is consistent with the dict if its variable is not already bound, OR if its
        variable is already bound to its argument.

        :param variable: ``Variable`` The variable to bind
        :param binding: ``Expression`` The atomic to which 'variable' should be bound
        :raise BindingException: If the variable cannot be bound in this dictionary
        """
        assert isinstance(variable, Variable)
        assert isinstance(binding, Expression)
        try:
            existing = self[variable]
        except KeyError:
            existing = None
        if not existing or binding == existing:
            self.d[variable] = binding
        elif isinstance(binding, IndividualVariableExpression):
            try:
                existing = self[binding.variable]
            except KeyError:
                existing = None
            binding2 = VariableExpression(variable)
            if not existing or binding2 == existing:
                self.d[binding.variable] = binding2
            else:
                raise BindingException('Variable %s already bound to another value' % variable)
        else:
            raise BindingException('Variable %s already bound to another value' % variable)

    def __getitem__(self, variable):
        """
        Return the expression to which 'variable' is bound
        """
        assert isinstance(variable, Variable)
        intermediate = self.d[variable]
        while intermediate:
            try:
                intermediate = self.d[intermediate]
            except KeyError:
                return intermediate

    def __contains__(self, item):
        return item in self.d

    def __add__(self, other):
        """
        :param other: ``BindingDict`` The dict with which to combine self
        :return: ``BindingDict`` A new dict containing all the elements of both parameters
        :raise BindingException: If the parameter dictionaries are not consistent with each other
        """
        try:
            combined = BindingDict()
            for v in self.d:
                combined[v] = self.d[v]
            for v in other.d:
                combined[v] = other.d[v]
            return combined
        except BindingException as e:
            raise BindingException("Attempting to add two contradicting BindingDicts: '%s' and '%s'" % (self, other)) from e

    def __len__(self):
        return len(self.d)

    def __str__(self):
        data_str = ', '.join((f'{v}: {self.d[v]}' for v in sorted(self.d.keys())))
        return '{' + data_str + '}'

    def __repr__(self):
        return '%s' % self