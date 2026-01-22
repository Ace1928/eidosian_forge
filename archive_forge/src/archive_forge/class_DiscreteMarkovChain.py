import random
import itertools
from typing import (Sequence as tSequence, Union as tUnion, List as tList,
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, Rational, igcd, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import gamma
from sympy.logic.boolalg import (And, Not, Or)
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.dense import (Matrix, eye, ones, zeros)
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.matrices.immutable import ImmutableMatrix
from sympy.sets.conditionset import ConditionSet
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Set, Union)
from sympy.solvers.solveset import linsolve
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import strongly_connected_components
from sympy.stats.joint_rv import JointDistribution
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.rv import (RandomIndexedSymbol, random_symbols, RandomSymbol,
from sympy.stats.stochastic_process import StochasticPSpace
from sympy.stats.symbolic_probability import Probability, Expectation
from sympy.stats.frv_types import Bernoulli, BernoulliDistribution, FiniteRV
from sympy.stats.drv_types import Poisson, PoissonDistribution
from sympy.stats.crv_types import Normal, NormalDistribution, Gamma, GammaDistribution
from sympy.core.sympify import _sympify, sympify
class DiscreteMarkovChain(DiscreteTimeStochasticProcess, MarkovProcess):
    """
    Represents a finite discrete time-homogeneous Markov chain.

    This type of Markov Chain can be uniquely characterised by
    its (ordered) state space and its one-step transition probability
    matrix.

    Parameters
    ==========

    sym:
        The name given to the Markov Chain
    state_space:
        Optional, by default, Range(n)
    trans_probs:
        Optional, by default, MatrixSymbol('_T', n, n)

    Examples
    ========

    >>> from sympy.stats import DiscreteMarkovChain, TransitionMatrixOf, P, E
    >>> from sympy import Matrix, MatrixSymbol, Eq, symbols
    >>> T = Matrix([[0.5, 0.2, 0.3],[0.2, 0.5, 0.3],[0.2, 0.3, 0.5]])
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    >>> YS = DiscreteMarkovChain("Y")

    >>> Y.state_space
    {0, 1, 2}
    >>> Y.transition_probabilities
    Matrix([
    [0.5, 0.2, 0.3],
    [0.2, 0.5, 0.3],
    [0.2, 0.3, 0.5]])
    >>> TS = MatrixSymbol('T', 3, 3)
    >>> P(Eq(YS[3], 2), Eq(YS[1], 1) & TransitionMatrixOf(YS, TS))
    T[0, 2]*T[1, 0] + T[1, 1]*T[1, 2] + T[1, 2]*T[2, 2]
    >>> P(Eq(Y[3], 2), Eq(Y[1], 1)).round(2)
    0.36

    Probabilities will be calculated based on indexes rather
    than state names. For example, with the Sunny-Cloudy-Rainy
    model with string state names:

    >>> from sympy.core.symbol import Str
    >>> Y = DiscreteMarkovChain("Y", [Str('Sunny'), Str('Cloudy'), Str('Rainy')], T)
    >>> P(Eq(Y[3], 2), Eq(Y[1], 1)).round(2)
    0.36

    This gives the same answer as the ``[0, 1, 2]`` state space.
    Currently, there is no support for state names within probability
    and expectation statements. Here is a work-around using ``Str``:

    >>> P(Eq(Str('Rainy'), Y[3]), Eq(Y[1], Str('Cloudy'))).round(2)
    0.36

    Symbol state names can also be used:

    >>> sunny, cloudy, rainy = symbols('Sunny, Cloudy, Rainy')
    >>> Y = DiscreteMarkovChain("Y", [sunny, cloudy, rainy], T)
    >>> P(Eq(Y[3], rainy), Eq(Y[1], cloudy)).round(2)
    0.36

    Expectations will be calculated as follows:

    >>> E(Y[3], Eq(Y[1], cloudy))
    0.38*Cloudy + 0.36*Rainy + 0.26*Sunny

    Probability of expressions with multiple RandomIndexedSymbols
    can also be calculated provided there is only 1 RandomIndexedSymbol
    in the given condition. It is always better to use Rational instead
    of floating point numbers for the probabilities in the
    transition matrix to avoid errors.

    >>> from sympy import Gt, Le, Rational
    >>> T = Matrix([[Rational(5, 10), Rational(3, 10), Rational(2, 10)], [Rational(2, 10), Rational(7, 10), Rational(1, 10)], [Rational(3, 10), Rational(3, 10), Rational(4, 10)]])
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    >>> P(Eq(Y[3], Y[1]), Eq(Y[0], 0)).round(3)
    0.409
    >>> P(Gt(Y[3], Y[1]), Eq(Y[0], 0)).round(2)
    0.36
    >>> P(Le(Y[15], Y[10]), Eq(Y[8], 2)).round(7)
    0.6963328

    Symbolic probability queries are also supported

    >>> a, b, c, d = symbols('a b c d')
    >>> T = Matrix([[Rational(1, 10), Rational(4, 10), Rational(5, 10)], [Rational(3, 10), Rational(4, 10), Rational(3, 10)], [Rational(7, 10), Rational(2, 10), Rational(1, 10)]])
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    >>> query = P(Eq(Y[a], b), Eq(Y[c], d))
    >>> query.subs({a:10, b:2, c:5, d:1}).round(4)
    0.3096
    >>> P(Eq(Y[10], 2), Eq(Y[5], 1)).evalf().round(4)
    0.3096
    >>> query_gt = P(Gt(Y[a], b), Eq(Y[c], d))
    >>> query_gt.subs({a:21, b:0, c:5, d:0}).evalf().round(5)
    0.64705
    >>> P(Gt(Y[21], 0), Eq(Y[5], 0)).round(5)
    0.64705

    There is limited support for arbitrarily sized states:

    >>> n = symbols('n', nonnegative=True, integer=True)
    >>> T = MatrixSymbol('T', n, n)
    >>> Y = DiscreteMarkovChain("Y", trans_probs=T)
    >>> Y.state_space
    Range(0, n, 1)
    >>> query = P(Eq(Y[a], b), Eq(Y[c], d))
    >>> query.subs({a:10, b:2, c:5, d:1})
    (T**5)[1, 2]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Markov_chain#Discrete-time_Markov_chain
    .. [2] https://web.archive.org/web/20201230182007/https://www.dartmouth.edu/~chance/teaching_aids/books_articles/probability_book/Chapter11.pdf
    """
    index_set = S.Naturals0

    def __new__(cls, sym, state_space=None, trans_probs=None):
        sym = _symbol_converter(sym)
        state_space, trans_probs = MarkovProcess._sanity_checks(state_space, trans_probs)
        obj = Basic.__new__(cls, sym, state_space, trans_probs)
        indices = {}
        if isinstance(obj.number_of_states, Integer):
            for index, state in enumerate(obj._state_index):
                indices[state] = index
        obj.index_of = indices
        return obj

    @property
    def transition_probabilities(self):
        """
        Transition probabilities of discrete Markov chain,
        either an instance of Matrix or MatrixSymbol.
        """
        return self.args[2]

    def communication_classes(self) -> tList[tTuple[tList[Basic], Boolean, Integer]]:
        """
        Returns the list of communication classes that partition
        the states of the markov chain.

        A communication class is defined to be a set of states
        such that every state in that set is reachable from
        every other state in that set. Due to its properties
        this forms a class in the mathematical sense.
        Communication classes are also known as recurrence
        classes.

        Returns
        =======

        classes
            The ``classes`` are a list of tuples. Each
            tuple represents a single communication class
            with its properties. The first element in the
            tuple is the list of states in the class, the
            second element is whether the class is recurrent
            and the third element is the period of the
            communication class.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix
        >>> T = Matrix([[0, 1, 0],
        ...             [1, 0, 0],
        ...             [1, 0, 0]])
        >>> X = DiscreteMarkovChain('X', [1, 2, 3], T)
        >>> classes = X.communication_classes()
        >>> for states, is_recurrent, period in classes:
        ...     states, is_recurrent, period
        ([1, 2], True, 2)
        ([3], False, 1)

        From this we can see that states ``1`` and ``2``
        communicate, are recurrent and have a period
        of 2. We can also see state ``3`` is transient
        with a period of 1.

        Notes
        =====

        The algorithm used is of order ``O(n**2)`` where
        ``n`` is the number of states in the markov chain.
        It uses Tarjan's algorithm to find the classes
        themselves and then it uses a breadth-first search
        algorithm to find each class's periodicity.
        Most of the algorithm's components approach ``O(n)``
        as the matrix becomes more and more sparse.

        References
        ==========

        .. [1] https://web.archive.org/web/20220207032113/https://www.columbia.edu/~ww2040/4701Sum07/4701-06-Notes-MCII.pdf
        .. [2] https://cecas.clemson.edu/~shierd/Shier/markov.pdf
        .. [3] https://ujcontent.uj.ac.za/esploro/outputs/graduate/Markov-chains--a-graph-theoretical/999849107691#file-0
        .. [4] https://www.mathworks.com/help/econ/dtmc.classify.html
        """
        n = self.number_of_states
        T = self.transition_probabilities
        if isinstance(T, MatrixSymbol):
            raise NotImplementedError('Cannot perform the operation with a symbolic matrix.')
        V = Range(n)
        E = [(i, j) for i in V for j in V if T[i, j] != 0]
        classes = strongly_connected_components((V, E))
        recurrence = []
        periods = []
        for class_ in classes:
            submatrix = T[class_, class_]
            is_recurrent = S.true
            rows = submatrix.tolist()
            for row in rows:
                if sum(row) - 1 != 0:
                    is_recurrent = S.false
                    break
            recurrence.append(is_recurrent)
            non_tree_edge_values: tSet[int] = set()
            visited = {class_[0]}
            newly_visited = {class_[0]}
            level = {class_[0]: 0}
            current_level = 0
            done = False
            while not done:
                done = len(visited) == len(class_)
                current_level += 1
                for i in newly_visited:
                    newly_visited = {j for j in class_ if T[i, j] != 0}
                    new_tree_edges = newly_visited.difference(visited)
                    for j in new_tree_edges:
                        level[j] = current_level
                    new_non_tree_edges = newly_visited.intersection(visited)
                    new_non_tree_edge_values = {level[i] - level[j] + 1 for j in new_non_tree_edges}
                    non_tree_edge_values = non_tree_edge_values.union(new_non_tree_edge_values)
                    visited = visited.union(new_tree_edges)
            positive_ntev = {val_e for val_e in non_tree_edge_values if val_e > 0}
            if len(positive_ntev) == 0:
                periods.append(len(class_))
            elif len(positive_ntev) == 1:
                periods.append(positive_ntev.pop())
            else:
                periods.append(igcd(*positive_ntev))
        classes = [[_sympify(self._state_index[i]) for i in class_] for class_ in classes]
        return list(zip(classes, recurrence, map(Integer, periods)))

    def fundamental_matrix(self):
        """
        Each entry fundamental matrix can be interpreted as
        the expected number of times the chains is in state j
        if it started in state i.

        References
        ==========

        .. [1] https://lips.cs.princeton.edu/the-fundamental-matrix-of-a-finite-markov-chain/

        """
        _, _, _, Q = self.decompose()
        if Q.shape[0] > 0:
            I = eye(Q.shape[0])
            if (I - Q).det() == 0:
                raise ValueError("The fundamental matrix doesn't exist.")
            return (I - Q).inv().as_immutable()
        else:
            P = self.transition_probabilities
            I = eye(P.shape[0])
            w = self.fixed_row_vector()
            W = Matrix([list(w) for i in range(0, P.shape[0])])
            if (I - P + W).det() == 0:
                raise ValueError("The fundamental matrix doesn't exist.")
            return (I - P + W).inv().as_immutable()

    def absorbing_probabilities(self):
        """
        Computes the absorbing probabilities, i.e.
        the ij-th entry of the matrix denotes the
        probability of Markov chain being absorbed
        in state j starting from state i.
        """
        _, _, R, _ = self.decompose()
        N = self.fundamental_matrix()
        if R is None or N is None:
            return None
        return N * R

    def absorbing_probabilites(self):
        sympy_deprecation_warning('\n            DiscreteMarkovChain.absorbing_probabilites() is deprecated. Use\n            absorbing_probabilities() instead (note the spelling difference).\n            ', deprecated_since_version='1.7', active_deprecations_target='deprecated-absorbing_probabilites')
        return self.absorbing_probabilities()

    def is_regular(self):
        tuples = self.communication_classes()
        if len(tuples) == 0:
            return S.false
        classes, _, periods = list(zip(*tuples))
        return And(len(classes) == 1, periods[0] == 1)

    def is_ergodic(self):
        tuples = self.communication_classes()
        if len(tuples) == 0:
            return S.false
        classes, _, _ = list(zip(*tuples))
        return S(len(classes) == 1)

    def is_absorbing_state(self, state):
        trans_probs = self.transition_probabilities
        if isinstance(trans_probs, ImmutableMatrix) and state < trans_probs.shape[0]:
            return S(trans_probs[state, state]) is S.One

    def is_absorbing_chain(self):
        states, A, B, C = self.decompose()
        r = A.shape[0]
        return And(r > 0, A == Identity(r).as_explicit())

    def stationary_distribution(self, condition_set=False) -> tUnion[ImmutableMatrix, ConditionSet, Lambda]:
        """
        The stationary distribution is any row vector, p, that solves p = pP,
        is row stochastic and each element in p must be nonnegative.
        That means in matrix form: :math:`(P-I)^T p^T = 0` and
        :math:`(1, \\dots, 1) p = 1`
        where ``P`` is the one-step transition matrix.

        All time-homogeneous Markov Chains with a finite state space
        have at least one stationary distribution. In addition, if
        a finite time-homogeneous Markov Chain is irreducible, the
        stationary distribution is unique.

        Parameters
        ==========

        condition_set : bool
            If the chain has a symbolic size or transition matrix,
            it will return a ``Lambda`` if ``False`` and return a
            ``ConditionSet`` if ``True``.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix, S

        An irreducible Markov Chain

        >>> T = Matrix([[S(1)/2, S(1)/2, 0],
        ...             [S(4)/5, S(1)/5, 0],
        ...             [1, 0, 0]])
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> X.stationary_distribution()
        Matrix([[8/13, 5/13, 0]])

        A reducible Markov Chain

        >>> T = Matrix([[S(1)/2, S(1)/2, 0],
        ...             [S(4)/5, S(1)/5, 0],
        ...             [0, 0, 1]])
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> X.stationary_distribution()
        Matrix([[8/13 - 8*tau0/13, 5/13 - 5*tau0/13, tau0]])

        >>> Y = DiscreteMarkovChain('Y')
        >>> Y.stationary_distribution()
        Lambda((wm, _T), Eq(wm*_T, wm))

        >>> Y.stationary_distribution(condition_set=True)
        ConditionSet(wm, Eq(wm*_T, wm))

        References
        ==========

        .. [1] https://www.probabilitycourse.com/chapter11/11_2_6_stationary_and_limiting_distributions.php
        .. [2] https://web.archive.org/web/20210508104430/https://galton.uchicago.edu/~yibi/teaching/stat317/2014/Lectures/Lecture4_6up.pdf

        See Also
        ========

        sympy.stats.DiscreteMarkovChain.limiting_distribution
        """
        trans_probs = self.transition_probabilities
        n = self.number_of_states
        if n == 0:
            return ImmutableMatrix(Matrix([[]]))
        if isinstance(trans_probs, MatrixSymbol):
            wm = MatrixSymbol('wm', 1, n)
            if condition_set:
                return ConditionSet(wm, Eq(wm * trans_probs, wm))
            else:
                return Lambda((wm, trans_probs), Eq(wm * trans_probs, wm))
        a = Matrix(trans_probs - Identity(n)).T
        a[0, 0:n] = ones(1, n)
        b = zeros(n, 1)
        b[0, 0] = 1
        soln = list(linsolve((a, b)))[0]
        return ImmutableMatrix([soln])

    def fixed_row_vector(self):
        """
        A wrapper for ``stationary_distribution()``.
        """
        return self.stationary_distribution()

    @property
    def limiting_distribution(self):
        """
        The fixed row vector is the limiting
        distribution of a discrete Markov chain.
        """
        return self.fixed_row_vector()

    def decompose(self) -> tTuple[tList[Basic], ImmutableMatrix, ImmutableMatrix, ImmutableMatrix]:
        """
        Decomposes the transition matrix into submatrices with
        special properties.

        The transition matrix can be decomposed into 4 submatrices:
        - A - the submatrix from recurrent states to recurrent states.
        - B - the submatrix from transient to recurrent states.
        - C - the submatrix from transient to transient states.
        - O - the submatrix of zeros for recurrent to transient states.

        Returns
        =======

        states, A, B, C
            ``states`` - a list of state names with the first being
            the recurrent states and the last being
            the transient states in the order
            of the row names of A and then the row names of C.
            ``A`` - the submatrix from recurrent states to recurrent states.
            ``B`` - the submatrix from transient to recurrent states.
            ``C`` - the submatrix from transient to transient states.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix, S

        One can decompose this chain for example:

        >>> T = Matrix([[S(1)/2, S(1)/2, 0,      0,      0],
        ...             [S(2)/5, S(1)/5, S(2)/5, 0,      0],
        ...             [0,      0,      1,      0,      0],
        ...             [0,      0,      S(1)/2, S(1)/2, 0],
        ...             [S(1)/2, 0,      0,      0, S(1)/2]])
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> states, A, B, C = X.decompose()
        >>> states
        [2, 0, 1, 3, 4]

        >>> A   # recurrent to recurrent
        Matrix([[1]])

        >>> B  # transient to recurrent
        Matrix([
        [  0],
        [2/5],
        [1/2],
        [  0]])

        >>> C  # transient to transient
        Matrix([
        [1/2, 1/2,   0,   0],
        [2/5, 1/5,   0,   0],
        [  0,   0, 1/2,   0],
        [1/2,   0,   0, 1/2]])

        This means that state 2 is the only absorbing state
        (since A is a 1x1 matrix). B is a 4x1 matrix since
        the 4 remaining transient states all merge into reccurent
        state 2. And C is the 4x4 matrix that shows how the
        transient states 0, 1, 3, 4 all interact.

        See Also
        ========

        sympy.stats.DiscreteMarkovChain.communication_classes
        sympy.stats.DiscreteMarkovChain.canonical_form

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Absorbing_Markov_chain
        .. [2] https://people.brandeis.edu/~igusa/Math56aS08/Math56a_S08_notes015.pdf
        """
        trans_probs = self.transition_probabilities
        classes = self.communication_classes()
        r_states = []
        t_states = []
        for states, recurrent, period in classes:
            if recurrent:
                r_states += states
            else:
                t_states += states
        states = r_states + t_states
        indexes = [self.index_of[state] for state in states]
        A = Matrix(len(r_states), len(r_states), lambda i, j: trans_probs[indexes[i], indexes[j]])
        B = Matrix(len(t_states), len(r_states), lambda i, j: trans_probs[indexes[len(r_states) + i], indexes[j]])
        C = Matrix(len(t_states), len(t_states), lambda i, j: trans_probs[indexes[len(r_states) + i], indexes[len(r_states) + j]])
        return (states, A.as_immutable(), B.as_immutable(), C.as_immutable())

    def canonical_form(self) -> tTuple[tList[Basic], ImmutableMatrix]:
        """
        Reorders the one-step transition matrix
        so that recurrent states appear first and transient
        states appear last. Other representations include inserting
        transient states first and recurrent states last.

        Returns
        =======

        states, P_new
            ``states`` is the list that describes the order of the
            new states in the matrix
            so that the ith element in ``states`` is the state of the
            ith row of A.
            ``P_new`` is the new transition matrix in canonical form.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix, S

        You can convert your chain into canonical form:

        >>> T = Matrix([[S(1)/2, S(1)/2, 0,      0,      0],
        ...             [S(2)/5, S(1)/5, S(2)/5, 0,      0],
        ...             [0,      0,      1,      0,      0],
        ...             [0,      0,      S(1)/2, S(1)/2, 0],
        ...             [S(1)/2, 0,      0,      0, S(1)/2]])
        >>> X = DiscreteMarkovChain('X', list(range(1, 6)), trans_probs=T)
        >>> states, new_matrix = X.canonical_form()
        >>> states
        [3, 1, 2, 4, 5]

        >>> new_matrix
        Matrix([
        [  1,   0,   0,   0,   0],
        [  0, 1/2, 1/2,   0,   0],
        [2/5, 2/5, 1/5,   0,   0],
        [1/2,   0,   0, 1/2,   0],
        [  0, 1/2,   0,   0, 1/2]])

        The new states are [3, 1, 2, 4, 5] and you can
        create a new chain with this and its canonical
        form will remain the same (since it is already
        in canonical form).

        >>> X = DiscreteMarkovChain('X', states, new_matrix)
        >>> states, new_matrix = X.canonical_form()
        >>> states
        [3, 1, 2, 4, 5]

        >>> new_matrix
        Matrix([
        [  1,   0,   0,   0,   0],
        [  0, 1/2, 1/2,   0,   0],
        [2/5, 2/5, 1/5,   0,   0],
        [1/2,   0,   0, 1/2,   0],
        [  0, 1/2,   0,   0, 1/2]])

        This is not limited to absorbing chains:

        >>> T = Matrix([[0, 5,  5, 0,  0],
        ...             [0, 0,  0, 10, 0],
        ...             [5, 0,  5, 0,  0],
        ...             [0, 10, 0, 0,  0],
        ...             [0, 3,  0, 3,  4]])/10
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> states, new_matrix = X.canonical_form()
        >>> states
        [1, 3, 0, 2, 4]

        >>> new_matrix
        Matrix([
        [   0,    1,   0,   0,   0],
        [   1,    0,   0,   0,   0],
        [ 1/2,    0,   0, 1/2,   0],
        [   0,    0, 1/2, 1/2,   0],
        [3/10, 3/10,   0,   0, 2/5]])

        See Also
        ========

        sympy.stats.DiscreteMarkovChain.communication_classes
        sympy.stats.DiscreteMarkovChain.decompose

        References
        ==========

        .. [1] https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470316887.app1
        .. [2] http://www.columbia.edu/~ww2040/6711F12/lect1023big.pdf
        """
        states, A, B, C = self.decompose()
        O = zeros(A.shape[0], C.shape[1])
        return (states, BlockMatrix([[A, O], [B, C]]).as_explicit())

    def sample(self):
        """
        Returns
        =======

        sample: iterator object
            iterator object containing the sample

        """
        if not isinstance(self.transition_probabilities, (Matrix, ImmutableMatrix)):
            raise ValueError('Transition Matrix must be provided for sampling')
        Tlist = self.transition_probabilities.tolist()
        samps = [random.choice(list(self.state_space))]
        yield samps[0]
        time = 1
        densities = {}
        for state in self.state_space:
            states = list(self.state_space)
            densities[state] = {states[i]: Tlist[state][i] for i in range(len(states))}
        while time < S.Infinity:
            samps.append(next(sample_iter(FiniteRV('_', densities[samps[time - 1]]))))
            yield samps[time]
            time += 1