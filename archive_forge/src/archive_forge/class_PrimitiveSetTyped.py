import copy
import math
import copyreg
import random
import re
import sys
import types
import warnings
from collections import defaultdict, deque
from functools import partial, wraps
from operator import eq, lt
from . import tools  # Needed by HARM-GP
class PrimitiveSetTyped(object):
    """Class that contains the primitives that can be used to solve a
    Strongly Typed GP problem. The set also defined the researched
    function return type, and input arguments type and number.
    """

    def __init__(self, name, in_types, ret_type, prefix='ARG'):
        self.terminals = defaultdict(list)
        self.primitives = defaultdict(list)
        self.arguments = []
        self.context = {'__builtins__': None}
        self.mapping = dict()
        self.terms_count = 0
        self.prims_count = 0
        self.name = name
        self.ret = ret_type
        self.ins = in_types
        for i, type_ in enumerate(in_types):
            arg_str = '{prefix}{index}'.format(prefix=prefix, index=i)
            self.arguments.append(arg_str)
            term = Terminal(arg_str, True, type_)
            self._add(term)
            self.terms_count += 1

    def renameArguments(self, **kargs):
        """Rename function arguments with new names from *kargs*.
        """
        for i, old_name in enumerate(self.arguments):
            if old_name in kargs:
                new_name = kargs[old_name]
                self.arguments[i] = new_name
                self.mapping[new_name] = self.mapping[old_name]
                self.mapping[new_name].value = new_name
                del self.mapping[old_name]

    def _add(self, prim):

        def addType(dict_, ret_type):
            if ret_type not in dict_:
                new_list = []
                for type_, list_ in dict_.items():
                    if issubclass(type_, ret_type):
                        for item in list_:
                            if item not in new_list:
                                new_list.append(item)
                dict_[ret_type] = new_list
        addType(self.primitives, prim.ret)
        addType(self.terminals, prim.ret)
        self.mapping[prim.name] = prim
        if isinstance(prim, Primitive):
            for type_ in prim.args:
                addType(self.primitives, type_)
                addType(self.terminals, type_)
            dict_ = self.primitives
        else:
            dict_ = self.terminals
        for type_ in dict_:
            if issubclass(prim.ret, type_):
                dict_[type_].append(prim)

    def addPrimitive(self, primitive, in_types, ret_type, name=None):
        """Add a primitive to the set.

        :param primitive: callable object or a function.
        :param in_types: list of primitives arguments' type
        :param ret_type: type returned by the primitive.
        :param name: alternative name for the primitive instead
                     of its __name__ attribute.
        """
        if name is None:
            name = primitive.__name__
        prim = Primitive(name, in_types, ret_type)
        assert name not in self.context or self.context[name] is primitive, "Primitives are required to have a unique name. Consider using the argument 'name' to rename your second '%s' primitive." % (name,)
        self._add(prim)
        self.context[prim.name] = primitive
        self.prims_count += 1

    def addTerminal(self, terminal, ret_type, name=None):
        """Add a terminal to the set. Terminals can be named
        using the optional *name* argument. This should be
        used : to define named constant (i.e.: pi); to speed the
        evaluation time when the object is long to build; when
        the object does not have a __repr__ functions that returns
        the code to build the object; when the object class is
        not a Python built-in.

        :param terminal: Object, or a function with no arguments.
        :param ret_type: Type of the terminal.
        :param name: defines the name of the terminal in the expression.
        """
        symbolic = False
        if name is None and callable(terminal):
            name = terminal.__name__
        assert name not in self.context, "Terminals are required to have a unique name. Consider using the argument 'name' to rename your second %s terminal." % (name,)
        if name is not None:
            self.context[name] = terminal
            terminal = name
            symbolic = True
        elif terminal in (True, False):
            self.context[str(terminal)] = terminal
        prim = Terminal(terminal, symbolic, ret_type)
        self._add(prim)
        self.terms_count += 1

    def addEphemeralConstant(self, name, ephemeral, ret_type):
        """Add an ephemeral constant to the set. An ephemeral constant
        is a no argument function that returns a random value. The value
        of the constant is constant for a Tree, but may differ from one
        Tree to another.

        :param name: name used to refers to this ephemeral type.
        :param ephemeral: function with no arguments returning a random value.
        :param ret_type: type of the object returned by *ephemeral*.
        """
        if not name in self.mapping:
            class_ = MetaEphemeral(name, ephemeral, ret_type)
        else:
            class_ = self.mapping[name]
            if class_.func is not ephemeral:
                raise Exception('Ephemerals with different functions should be named differently, even between psets.')
            if class_.ret is not ret_type:
                raise Exception('Ephemerals with the same name and function should have the same type, even between psets.')
        self._add(class_)
        self.terms_count += 1

    def addADF(self, adfset):
        """Add an Automatically Defined Function (ADF) to the set.

        :param adfset: PrimitiveSetTyped containing the primitives with which
                       the ADF can be built.
        """
        prim = Primitive(adfset.name, adfset.ins, adfset.ret)
        self._add(prim)
        self.prims_count += 1

    @property
    def terminalRatio(self):
        """Return the ratio of the number of terminals on the number of all
        kind of primitives.
        """
        return self.terms_count / float(self.terms_count + self.prims_count)