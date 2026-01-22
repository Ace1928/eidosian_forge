from __future__ import print_function
import warnings
import numbers
import six
import numpy as np
from patsy import PatsyError
from patsy.util import atleast_2d_column_default
from patsy.compat import OrderedDict
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
from patsy.constraint import linear_constraint
from patsy.contrasts import ContrastMatrix
from patsy.desc import ModelDesc, Term
class DesignInfo(object):
    """A DesignInfo object holds metadata about a design matrix.

    This is the main object that Patsy uses to pass metadata about a design
    matrix to statistical libraries, in order to allow further downstream
    processing like intelligent tests, prediction on new data, etc. Usually
    encountered as the `.design_info` attribute on design matrices.

    """

    def __init__(self, column_names, factor_infos=None, term_codings=None):
        self.column_name_indexes = OrderedDict(zip(column_names, range(len(column_names))))
        if (factor_infos is None) != (term_codings is None):
            raise ValueError('Must specify either both or neither of factor_infos= and term_codings=')
        self.factor_infos = factor_infos
        self.term_codings = term_codings
        if self.factor_infos is not None:
            if not isinstance(self.factor_infos, dict):
                raise ValueError('factor_infos should be a dict')
            if not isinstance(self.term_codings, OrderedDict):
                raise ValueError('term_codings must be an OrderedDict')
            for term, subterms in six.iteritems(self.term_codings):
                if not isinstance(term, Term):
                    raise ValueError('expected a Term, not %r' % (term,))
                if not isinstance(subterms, list):
                    raise ValueError('term_codings must contain lists')
                term_factors = set(term.factors)
                for subterm in subterms:
                    if not isinstance(subterm, SubtermInfo):
                        raise ValueError('expected SubtermInfo, not %r' % (subterm,))
                    if not term_factors.issuperset(subterm.factors):
                        raise ValueError('unexpected factors in subterm')
            all_factors = set()
            for term in self.term_codings:
                all_factors.update(term.factors)
            if all_factors != set(self.factor_infos):
                raise ValueError('Provided Term objects and factor_infos do not match')
            for factor, factor_info in six.iteritems(self.factor_infos):
                if not isinstance(factor_info, FactorInfo):
                    raise ValueError('expected FactorInfo object, not %r' % (factor_info,))
                if factor != factor_info.factor:
                    raise ValueError('mismatched factor_info.factor')
            for term, subterms in six.iteritems(self.term_codings):
                for subterm in subterms:
                    exp_cols = 1
                    cat_factors = set()
                    for factor in subterm.factors:
                        fi = self.factor_infos[factor]
                        if fi.type == 'numerical':
                            exp_cols *= fi.num_columns
                        else:
                            assert fi.type == 'categorical'
                            cm = subterm.contrast_matrices[factor].matrix
                            if cm.shape[0] != len(fi.categories):
                                raise ValueError('Mismatched contrast matrix for factor %r' % (factor,))
                            cat_factors.add(factor)
                            exp_cols *= cm.shape[1]
                    if cat_factors != set(subterm.contrast_matrices):
                        raise ValueError('Mismatch between contrast_matrices and categorical factors')
                    if exp_cols != subterm.num_columns:
                        raise ValueError('Unexpected num_columns')
        if term_codings is None:
            self.term_slices = None
            term_names = column_names
            slices = [slice(i, i + 1) for i in range(len(column_names))]
            self.term_name_slices = OrderedDict(zip(term_names, slices))
        else:
            self.term_slices = OrderedDict()
            idx = 0
            for term, subterm_infos in six.iteritems(self.term_codings):
                term_columns = 0
                for subterm_info in subterm_infos:
                    term_columns += subterm_info.num_columns
                self.term_slices[term] = slice(idx, idx + term_columns)
                idx += term_columns
            if idx != len(self.column_names):
                raise ValueError('mismatch between column_names and columns coded by given terms')
            self.term_name_slices = OrderedDict([(term.name(), slice_) for term, slice_ in six.iteritems(self.term_slices)])
        assert self.term_name_slices is not None
        if self.term_slices is not None:
            assert list(self.term_slices.values()) == list(self.term_name_slices.values())
        covered = 0
        for slice_ in six.itervalues(self.term_name_slices):
            start, stop, step = slice_.indices(len(column_names))
            assert start == covered
            assert step == 1
            covered = stop
        assert covered == len(column_names)
        for column_name, index in six.iteritems(self.column_name_indexes):
            if column_name in self.term_name_slices:
                slice_ = self.term_name_slices[column_name]
                if slice_ != slice(index, index + 1):
                    raise ValueError('term/column name collision')
    __repr__ = repr_pretty_delegate

    def _repr_pretty_(self, p, cycle):
        assert not cycle
        repr_pretty_impl(p, self, [self.column_names], [('factor_infos', self.factor_infos), ('term_codings', self.term_codings)])

    @property
    def column_names(self):
        """A list of the column names, in order."""
        return list(self.column_name_indexes)

    @property
    def terms(self):
        """A list of :class:`Terms`, in order, or else None."""
        if self.term_slices is None:
            return None
        return list(self.term_slices)

    @property
    def term_names(self):
        """A list of terms, in order."""
        return list(self.term_name_slices)

    @property
    def builder(self):
        """.. deprecated:: 0.4.0"""
        warnings.warn(DeprecationWarning("The DesignInfo.builder attribute is deprecated starting in patsy v0.4.0; distinct builder objects have been eliminated and design_info.builder is now just a long-winded way of writing 'design_info' (i.e. the .builder attribute just returns self)"), stacklevel=2)
        return self

    @property
    def design_info(self):
        """.. deprecated:: 0.4.0"""
        warnings.warn(DeprecationWarning("Starting in patsy v0.4.0, the DesignMatrixBuilder class has been merged into the DesignInfo class. So there's no need to use builder.design_info to access the DesignInfo; 'builder' already *is* a DesignInfo."), stacklevel=2)
        return self

    def slice(self, columns_specifier):
        """Locate a subset of design matrix columns, specified symbolically.

        A patsy design matrix has two levels of structure: the individual
        columns (which are named), and the :ref:`terms <formulas>` in
        the formula that generated those columns. This is a one-to-many
        relationship: a single term may span several columns. This method
        provides a user-friendly API for locating those columns.

        (While we talk about columns here, this is probably most useful for
        indexing into other arrays that are derived from the design matrix,
        such as regression coefficients or covariance matrices.)

        The `columns_specifier` argument can take a number of forms:

        * A term name
        * A column name
        * A :class:`Term` object
        * An integer giving a raw index
        * A raw slice object

        In all cases, a Python :func:`slice` object is returned, which can be
        used directly for indexing.

        Example::

          y, X = dmatrices("y ~ a", demo_data("y", "a", nlevels=3))
          betas = np.linalg.lstsq(X, y)[0]
          a_betas = betas[X.design_info.slice("a")]

        (If you want to look up a single individual column by name, use
        ``design_info.column_name_indexes[name]``.)
        """
        if isinstance(columns_specifier, slice):
            return columns_specifier
        if np.issubdtype(type(columns_specifier), np.integer):
            return slice(columns_specifier, columns_specifier + 1)
        if self.term_slices is not None and columns_specifier in self.term_slices:
            return self.term_slices[columns_specifier]
        if columns_specifier in self.term_name_slices:
            return self.term_name_slices[columns_specifier]
        if columns_specifier in self.column_name_indexes:
            idx = self.column_name_indexes[columns_specifier]
            return slice(idx, idx + 1)
        raise PatsyError("unknown column specified '%s'" % (columns_specifier,))

    def linear_constraint(self, constraint_likes):
        """Construct a linear constraint in matrix form from a (possibly
        symbolic) description.

        Possible inputs:

        * A dictionary which is taken as a set of equality constraint. Keys
          can be either string column names, or integer column indexes.
        * A string giving a arithmetic expression referring to the matrix
          columns by name.
        * A list of such strings which are ANDed together.
        * A tuple (A, b) where A and b are array_likes, and the constraint is
          Ax = b. If necessary, these will be coerced to the proper
          dimensionality by appending dimensions with size 1.

        The string-based language has the standard arithmetic operators, / * +
        - and parentheses, plus "=" is used for equality and "," is used to
        AND together multiple constraint equations within a string. You can
        If no = appears in some expression, then that expression is assumed to
        be equal to zero. Division is always float-based, even if
        ``__future__.true_division`` isn't in effect.

        Returns a :class:`LinearConstraint` object.

        Examples::

          di = DesignInfo(["x1", "x2", "x3"])

          # Equivalent ways to write x1 == 0:
          di.linear_constraint({"x1": 0})  # by name
          di.linear_constraint({0: 0})  # by index
          di.linear_constraint("x1 = 0")  # string based
          di.linear_constraint("x1")  # can leave out "= 0"
          di.linear_constraint("2 * x1 = (x1 + 2 * x1) / 3")
          di.linear_constraint(([1, 0, 0], 0))  # constraint matrices

          # Equivalent ways to write x1 == 0 and x3 == 10
          di.linear_constraint({"x1": 0, "x3": 10})
          di.linear_constraint({0: 0, 2: 10})
          di.linear_constraint({0: 0, "x3": 10})
          di.linear_constraint("x1 = 0, x3 = 10")
          di.linear_constraint("x1, x3 = 10")
          di.linear_constraint(["x1", "x3 = 0"])  # list of strings
          di.linear_constraint("x1 = 0, x3 - 10 = x1")
          di.linear_constraint([[1, 0, 0], [0, 0, 1]], [0, 10])

          # You can also chain together equalities, just like Python:
          di.linear_constraint("x1 = x2 = 3")
        """
        return linear_constraint(constraint_likes, self.column_names)

    def describe(self):
        """Returns a human-readable string describing this design info.

        Example:

        .. ipython::

          In [1]: y, X = dmatrices("y ~ x1 + x2", demo_data("y", "x1", "x2"))

          In [2]: y.design_info.describe()
          Out[2]: 'y'

          In [3]: X.design_info.describe()
          Out[3]: '1 + x1 + x2'

        .. warning::

           There is no guarantee that the strings returned by this function
           can be parsed as formulas, or that if they can be parsed as a
           formula that they will produce a model equivalent to the one you
           started with. This function produces a best-effort description
           intended for humans to read.

        """
        names = []
        for name in self.term_names:
            if name == 'Intercept':
                names.append('1')
            else:
                names.append(name)
        return ' + '.join(names)

    def subset(self, which_terms):
        """Create a new :class:`DesignInfo` for design matrices that contain a
        subset of the terms that the current :class:`DesignInfo` does.

        For example, if ``design_info`` has terms ``x``, ``y``, and ``z``,
        then::

          design_info2 = design_info.subset(["x", "z"])

        will return a new DesignInfo that can be used to construct design
        matrices with only the columns corresponding to the terms ``x`` and
        ``z``. After we do this, then in general these two expressions will
        return the same thing (here we assume that ``x``, ``y``, and ``z``
        each generate a single column of the output)::

          build_design_matrix([design_info], data)[0][:, [0, 2]]
          build_design_matrix([design_info2], data)[0]

        However, a critical difference is that in the second case, ``data``
        need not contain any values for ``y``. This is very useful when doing
        prediction using a subset of a model, in which situation R usually
        forces you to specify dummy values for ``y``.

        If using a formula to specify the terms to include, remember that like
        any formula, the intercept term will be included by default, so use
        ``0`` or ``-1`` in your formula if you want to avoid this.

        This method can also be used to reorder the terms in your design
        matrix, in case you want to do that for some reason. I can't think of
        any.

        Note that this method will generally *not* produce the same result as
        creating a new model directly. Consider these DesignInfo objects::

            design1 = dmatrix("1 + C(a)", data)
            design2 = design1.subset("0 + C(a)")
            design3 = dmatrix("0 + C(a)", data)

        Here ``design2`` and ``design3`` will both produce design matrices
        that contain an encoding of ``C(a)`` without any intercept term. But
        ``design3`` uses a full-rank encoding for the categorical term
        ``C(a)``, while ``design2`` uses the same reduced-rank encoding as
        ``design1``.

        :arg which_terms: The terms which should be kept in the new
          :class:`DesignMatrixBuilder`. If this is a string, then it is parsed
          as a formula, and then the names of the resulting terms are taken as
          the terms to keep. If it is a list, then it can contain a mixture of
          term names (as strings) and :class:`Term` objects.

        .. versionadded: 0.2.0
           New method on the class DesignMatrixBuilder.

        .. versionchanged: 0.4.0
           Moved from DesignMatrixBuilder to DesignInfo, as part of the
           removal of DesignMatrixBuilder.

        """
        if isinstance(which_terms, str):
            desc = ModelDesc.from_formula(which_terms)
            if desc.lhs_termlist:
                raise PatsyError('right-hand-side-only formula required')
            which_terms = [term.name() for term in desc.rhs_termlist]
        if self.term_codings is None:
            new_names = []
            for t in which_terms:
                new_names += self.column_names[self.term_name_slices[t]]
            return DesignInfo(new_names)
        else:
            term_name_to_term = {}
            for term in self.term_codings:
                term_name_to_term[term.name()] = term
            new_column_names = []
            new_factor_infos = {}
            new_term_codings = OrderedDict()
            for name_or_term in which_terms:
                term = term_name_to_term.get(name_or_term, name_or_term)
                s = self.term_slices[term]
                new_column_names += self.column_names[s]
                for f in term.factors:
                    new_factor_infos[f] = self.factor_infos[f]
                new_term_codings[term] = self.term_codings[term]
            return DesignInfo(new_column_names, factor_infos=new_factor_infos, term_codings=new_term_codings)

    @classmethod
    def from_array(cls, array_like, default_column_prefix='column'):
        """Find or construct a DesignInfo appropriate for a given array_like.

        If the input `array_like` already has a ``.design_info``
        attribute, then it will be returned. Otherwise, a new DesignInfo
        object will be constructed, using names either taken from the
        `array_like` (e.g., for a pandas DataFrame with named columns), or
        constructed using `default_column_prefix`.

        This is how :func:`dmatrix` (for example) creates a DesignInfo object
        if an arbitrary matrix is passed in.

        :arg array_like: An ndarray or pandas container.
        :arg default_column_prefix: If it's necessary to invent column names,
          then this will be used to construct them.
        :returns: a DesignInfo object
        """
        if hasattr(array_like, 'design_info') and isinstance(array_like.design_info, cls):
            return array_like.design_info
        arr = atleast_2d_column_default(array_like, preserve_pandas=True)
        if arr.ndim > 2:
            raise ValueError("design matrix can't have >2 dimensions")
        columns = getattr(arr, 'columns', range(arr.shape[1]))
        if hasattr(columns, 'dtype') and (not safe_issubdtype(columns.dtype, np.integer)):
            column_names = [str(obj) for obj in columns]
        else:
            column_names = ['%s%s' % (default_column_prefix, i) for i in columns]
        return DesignInfo(column_names)
    __getstate__ = no_pickling