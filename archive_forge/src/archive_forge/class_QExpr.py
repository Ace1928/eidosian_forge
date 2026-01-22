from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.core.containers import Tuple
from sympy.utilities.iterables import is_sequence
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.matrixutils import (
class QExpr(Expr):
    """A base class for all quantum object like operators and states."""
    __slots__ = ('hilbert_space',)
    is_commutative = False
    _label_separator = ''

    @property
    def free_symbols(self):
        return {self}

    def __new__(cls, *args, **kwargs):
        """Construct a new quantum object.

        Parameters
        ==========

        args : tuple
            The list of numbers or parameters that uniquely specify the
            quantum object. For a state, this will be its symbol or its
            set of quantum numbers.

        Examples
        ========

        >>> from sympy.physics.quantum.qexpr import QExpr
        >>> q = QExpr(0)
        >>> q
        0
        >>> q.label
        (0,)
        >>> q.hilbert_space
        H
        >>> q.args
        (0,)
        >>> q.is_commutative
        False
        """
        args = cls._eval_args(args, **kwargs)
        if len(args) == 0:
            args = cls._eval_args(tuple(cls.default_args()), **kwargs)
        inst = Expr.__new__(cls, *args)
        inst.hilbert_space = cls._eval_hilbert_space(args)
        return inst

    @classmethod
    def _new_rawargs(cls, hilbert_space, *args, **old_assumptions):
        """Create new instance of this class with hilbert_space and args.

        This is used to bypass the more complex logic in the ``__new__``
        method in cases where you already have the exact ``hilbert_space``
        and ``args``. This should be used when you are positive these
        arguments are valid, in their final, proper form and want to optimize
        the creation of the object.
        """
        obj = Expr.__new__(cls, *args, **old_assumptions)
        obj.hilbert_space = hilbert_space
        return obj

    @property
    def label(self):
        """The label is the unique set of identifiers for the object.

        Usually, this will include all of the information about the state
        *except* the time (in the case of time-dependent objects).

        This must be a tuple, rather than a Tuple.
        """
        if len(self.args) == 0:
            return self._eval_args(list(self.default_args()))
        else:
            return self.args

    @property
    def is_symbolic(self):
        return True

    @classmethod
    def default_args(self):
        """If no arguments are specified, then this will return a default set
        of arguments to be run through the constructor.

        NOTE: Any classes that override this MUST return a tuple of arguments.
        Should be overridden by subclasses to specify the default arguments for kets and operators
        """
        raise NotImplementedError('No default arguments for this class!')

    def _eval_adjoint(self):
        obj = Expr._eval_adjoint(self)
        if obj is None:
            obj = Expr.__new__(Dagger, self)
        if isinstance(obj, QExpr):
            obj.hilbert_space = self.hilbert_space
        return obj

    @classmethod
    def _eval_args(cls, args):
        """Process the args passed to the __new__ method.

        This simply runs args through _qsympify_sequence.
        """
        return _qsympify_sequence(args)

    @classmethod
    def _eval_hilbert_space(cls, args):
        """Compute the Hilbert space instance from the args.
        """
        from sympy.physics.quantum.hilbert import HilbertSpace
        return HilbertSpace()

    def _print_sequence(self, seq, sep, printer, *args):
        result = []
        for item in seq:
            result.append(printer._print(item, *args))
        return sep.join(result)

    def _print_sequence_pretty(self, seq, sep, printer, *args):
        pform = printer._print(seq[0], *args)
        for item in seq[1:]:
            pform = prettyForm(*pform.right(sep))
            pform = prettyForm(*pform.right(printer._print(item, *args)))
        return pform

    def _print_subscript_pretty(self, a, b):
        top = prettyForm(*b.left(' ' * a.width()))
        bot = prettyForm(*a.right(' ' * b.width()))
        return prettyForm(*bot.below(top), binding=prettyForm.POW)

    def _print_superscript_pretty(self, a, b):
        return a ** b

    def _print_parens_pretty(self, pform, left='(', right=')'):
        return prettyForm(*pform.parens(left=left, right=right))

    def _print_label(self, printer, *args):
        """Prints the label of the QExpr

        This method prints self.label, using self._label_separator to separate
        the elements. This method should not be overridden, instead, override
        _print_contents to change printing behavior.
        """
        return self._print_sequence(self.label, self._label_separator, printer, *args)

    def _print_label_repr(self, printer, *args):
        return self._print_sequence(self.label, ',', printer, *args)

    def _print_label_pretty(self, printer, *args):
        return self._print_sequence_pretty(self.label, self._label_separator, printer, *args)

    def _print_label_latex(self, printer, *args):
        return self._print_sequence(self.label, self._label_separator, printer, *args)

    def _print_contents(self, printer, *args):
        """Printer for contents of QExpr

        Handles the printing of any unique identifying contents of a QExpr to
        print as its contents, such as any variables or quantum numbers. The
        default is to print the label, which is almost always the args. This
        should not include printing of any brackets or parentheses.
        """
        return self._print_label(printer, *args)

    def _print_contents_pretty(self, printer, *args):
        return self._print_label_pretty(printer, *args)

    def _print_contents_latex(self, printer, *args):
        return self._print_label_latex(printer, *args)

    def _sympystr(self, printer, *args):
        """Default printing behavior of QExpr objects

        Handles the default printing of a QExpr. To add other things to the
        printing of the object, such as an operator name to operators or
        brackets to states, the class should override the _print/_pretty/_latex
        functions directly and make calls to _print_contents where appropriate.
        This allows things like InnerProduct to easily control its printing the
        printing of contents.
        """
        return self._print_contents(printer, *args)

    def _sympyrepr(self, printer, *args):
        classname = self.__class__.__name__
        label = self._print_label_repr(printer, *args)
        return '%s(%s)' % (classname, label)

    def _pretty(self, printer, *args):
        pform = self._print_contents_pretty(printer, *args)
        return pform

    def _latex(self, printer, *args):
        return self._print_contents_latex(printer, *args)

    def _represent_default_basis(self, **options):
        raise NotImplementedError('This object does not have a default basis')

    def _represent(self, *, basis=None, **options):
        """Represent this object in a given basis.

        This method dispatches to the actual methods that perform the
        representation. Subclases of QExpr should define various methods to
        determine how the object will be represented in various bases. The
        format of these methods is::

            def _represent_BasisName(self, basis, **options):

        Thus to define how a quantum object is represented in the basis of
        the operator Position, you would define::

            def _represent_Position(self, basis, **options):

        Usually, basis object will be instances of Operator subclasses, but
        there is a chance we will relax this in the future to accommodate other
        types of basis sets that are not associated with an operator.

        If the ``format`` option is given it can be ("sympy", "numpy",
        "scipy.sparse"). This will ensure that any matrices that result from
        representing the object are returned in the appropriate matrix format.

        Parameters
        ==========

        basis : Operator
            The Operator whose basis functions will be used as the basis for
            representation.
        options : dict
            A dictionary of key/value pairs that give options and hints for
            the representation, such as the number of basis functions to
            be used.
        """
        if basis is None:
            result = self._represent_default_basis(**options)
        else:
            result = dispatch_method(self, '_represent', basis, **options)
        format = options.get('format', 'sympy')
        result = self._format_represent(result, format)
        return result

    def _format_represent(self, result, format):
        if format == 'sympy' and (not isinstance(result, Matrix)):
            return to_sympy(result)
        elif format == 'numpy' and (not isinstance(result, numpy_ndarray)):
            return to_numpy(result)
        elif format == 'scipy.sparse' and (not isinstance(result, scipy_sparse_matrix)):
            return to_scipy_sparse(result)
        return result