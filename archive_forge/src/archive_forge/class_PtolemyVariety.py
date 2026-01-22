from . import matrix
from . import homology
from .polynomial import Polynomial
from .ptolemyObstructionClass import PtolemyObstructionClass
from .ptolemyGeneralizedObstructionClass import PtolemyGeneralizedObstructionClass
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
from . import processFileBase, processFileDispatch, processMagmaFile
from . import utilities
from string import Template
import signal
import re
import os
import sys
from urllib.request import Request, urlopen
from urllib.request import quote as urlquote
from urllib.error import HTTPError
class PtolemyVariety:
    """
    Holds a reduced Ptolemy variety.

    === Examples ===

    To generate such a variety, call:

    >>> from snappy import Manifold
    >>> p = Manifold("4_1").ptolemy_variety(2, obstruction_class = 1)

    Show the equations and variables:

    >>> for e in p.equations: print(e)
    - c_0011_0 * c_0101_0 + c_0011_0^2 + c_0101_0^2
    c_0011_0 * c_0101_0 - c_0011_0^2 - c_0101_0^2
    - 1 + c_0011_0
    >>> p.variables
    ['c_0011_0', 'c_0101_0']

    Show as an ideal (sage object):

    >>> p.ideal    #doctest: +SKIP
    Ideal (-c_0011_0^2 + c_0011_0*c_0101_0 + c_0101_0^2, -c_0011_0^2 - c_0011_0*c_0101_0 + c_0101_0^2, c_0011_0 - 1) of Multivariate Polynomial Ring in c_0011_0, c_0101_0 over Rational Field
    (skip doctest because example only works in sage and not plain python)


    Produce magma input:

    >>> s = p.to_magma()
    >>> print(s.split('ring and ideal')[1].strip())    #doctest: +ELLIPSIS
    R<c_0011_0, c_0101_0> := PolynomialRing(RationalField(), 2, "grevlex");
    MyIdeal := ideal<R |
              - c_0011_0 * c_0101_0 + c_0011_0^2 + c_0101_0^2,
        ...

    Call ``p.compute_solutions()`` to automatically compute solutions!

    Show canonical representatives:

    (The Ptolemy coordinates c_0110_0 and c_0101_0 are identified, this
    information is needed to recover all Ptolemy coordinates from the solutions
    of a simplified Ptolemy variety. The information is also packaged into a
    python section by py_eval_variable_dict().)

    >>> p.canonical_representative["c_0110_0"]
    (1, 0, 'c_0101_0')

    """

    def __init__(self, manifold, N, obstruction_class, simplify, eliminate_fixed_ptolemys):
        self._manifold = manifold
        self._N = N
        self._obstruction_class = obstruction_class
        if obstruction_class:
            obstruction_class._checkManifoldAndN(manifold, N)
        if isinstance(obstruction_class, PtolemyObstructionClass):
            self._identified_variables_from_obstruction = obstruction_class.identified_variables
        else:
            self._identified_variables_from_obstruction = []
        H2_class = None
        if isinstance(obstruction_class, PtolemyGeneralizedObstructionClass):
            H2_class = obstruction_class.H2_class
        self._identified_coordinates = manifold._ptolemy_equations_identified_coordinates(N, H2_class)
        self._action_by_decoration_change = manifold._ptolemy_equations_action_by_decoration_change(N)
        self._fixed_ptolemy_coordinates = _fix_decoration(N, self._action_by_decoration_change)
        self._identified_variables = self._identified_coordinates + self._identified_variables_from_obstruction
        self._ptolemy_relations = _generate_ptolemy_relations(N, manifold.num_tetrahedra(), isinstance(obstruction_class, PtolemyObstructionClass))
        self.equations = [eqn for eqn in self._ptolemy_relations]
        order_of_u = 1
        if isinstance(obstruction_class, PtolemyGeneralizedObstructionClass):
            order_of_u, equations = obstruction_class._get_equation_for_u(N)
            self.equations += equations
        if eliminate_fixed_ptolemys:
            self._identified_variables += [(+1, 0, ptolemy_coord, 1) for ptolemy_coord in self._fixed_ptolemy_coordinates]
        else:
            one = Polynomial.constant_polynomial(1)
            self.equations += [Polynomial.from_variable_name(ptolemy_coord) - one for ptolemy_coord in self._fixed_ptolemy_coordinates]
        variables = _union([eqn.variables() for eqn in self.equations])
        if simplify:
            self.canonical_representative = _identified_variables_canonize(self._identified_variables)
            substitution = _canonical_representative_to_polynomial_substituition(self.canonical_representative, order_of_u)
            self.equations = [eqn.substitute(substitution) for eqn in self.equations]
        else:
            self.canonical_representative = {}
            for sign, power, var1, var2 in self._identified_variables:
                self.canonical_representative[var1] = (+1, 0, var1)
                if var2 != 1:
                    self.canonical_representative[var2] = (+1, 0, var2)
                if order_of_u == 2:
                    u = Polynomial.constant_polynomial(-1)
                else:
                    u = Polynomial.from_variable_name('u')
                firstTerm = Polynomial.from_variable_name(var1) * u ** (power % order_of_u)
                if var2 == 1:
                    secondTerm = Polynomial.constant_polynomial(sign)
                else:
                    secondTerm = Polynomial.constant_polynomial(sign) * Polynomial.from_variable_name(var2)
                self.equations.append(firstTerm - secondTerm)
        self.variables = _union([eqn.variables() for eqn in self.equations])
        for var in self.variables:
            if var[0:2] == 'c_':
                if var not in self.canonical_representative:
                    self.canonical_representative[var] = (+1, 0, var)
        self.variables_with_non_zero_condition = ['t'] + self.variables
        vars_without_u = [var for var in self.variables_with_non_zero_condition if not var == 'u']
        self._non_zero_condition = _non_zero_condition(vars_without_u)
        self.equations_with_non_zero_condition = self.equations + [self._non_zero_condition]
        if _within_sage:

            def sage_monomial(monomial):
                r = monomial.get_coefficient()
                for varName, expo in monomial.get_vars():
                    r = r * sageVariable(varName) ** expo
                return r

            def sage_eqn(eqn):
                return sum([sage_monomial(m) for m in eqn.get_monomials()])

            def sage_ideal(vars, eqns):
                polynomialRing = PolynomialRing(RationalField(), vars, order='lex')
                return Ideal(polynomialRing, [sage_eqn(eqn) for eqn in eqns])
            self.ideal = sage_ideal(self.variables, self.equations)
            self.ideal_with_non_zero_condition = sage_ideal(self.variables_with_non_zero_condition, self.equations_with_non_zero_condition)

    def py_eval_variable_dict(self):

        def create_dict_entry(var1, val):
            sign, power, var2 = val
            assert sign in [+1, -1]
            p = ''
            if self._N == 2:
                sign *= (-1) ** power
            elif power % self._N:
                p = " * d['u'] ** %d" % (power % self._N)
            if sign == +1:
                return "'%s' : d['%s']%s" % (var1, var2, p)
            else:
                return "'%s' : - d['%s']%s" % (var1, var2, p)
        format_str = '(lambda d: {\n          %s})'
        return format_str % ',\n          '.join([create_dict_entry(key, val) for key, val in list(self.canonical_representative.items()) if not key == 1])

    def py_eval_section(self):
        """
        Returns a string that can be evaluated in python and contains extra
        information needed to recover solutions from a simplified Ptolemy
        variety.

        >>> from snappy import Manifold, pari
        >>> M = Manifold('4_1')
        >>> p = M.ptolemy_variety(2, obstruction_class = 1)

        Get extra information

        >>> eval_section = p.py_eval_section()
        >>> print(eval_section)    #doctest: +ELLIPSIS
        {'variable_dict' :
             (lambda d: {
            ...

        Turn it into a python object by evaluation.

        >>> obj = eval(eval_section)

        Access the function that expands a solution to the simplified
        Ptolemy variety to a full solution.

        >>> variable_dict = obj['variable_dict']

        Setup a solution and expand it to a full solution, '1' must map to 1

        >>> simplified_solution = {'c_0101_0' : pari('0.5 - 0.866025403784439*I'), '1' : pari(1), 'c_0011_0' : pari(1)}
        >>> full_solution = variable_dict(simplified_solution)

        Full solution is a dictionary with a key for every Ptolemy coordinate

        >>> full_solution['c_1010_1']
        1
        >>> for tet in range(2):
        ...     for i in utilities.quadruples_with_fixed_sum_iterator(2, skipVertices = True):
        ...         c = "c_%d%d%d%d" % i + "_%d" % tet
        ...         assert c in full_solution
        """
        result = '{'
        result += "'variable_dict' :\n     %s" % self.py_eval_variable_dict()
        if isinstance(self._obstruction_class, PtolemyGeneralizedObstructionClass):
            if self._obstruction_class._is_non_trivial(self._N):
                result += ",\n 'non_trivial_generalized_obstruction_class' : True"
        result += '}'
        return result

    def to_magma_file(self, filename, template_path='magma/default.magma_template'):
        """
        >>> import os, tempfile
        >>> from snappy import Manifold
        >>> handle, name = tempfile.mkstemp()
        >>> p = Manifold("4_1").ptolemy_variety(2, obstruction_class=1)
        >>> p.to_magma_file(name)
        >>> os.close(handle); os.remove(name)
        """
        with open(filename, 'wb') as output:
            output.write(bytes(self.to_magma(template_path=template_path).encode('ascii')))

    def to_magma(self, template_path='magma/default.magma_template'):
        """
        Returns a string with the ideal that can be used as input for magma.

        The advanced user can provide a template string to write own magma
        code to process the equations.

        >>> from snappy import *
        >>> p = Manifold("4_1").ptolemy_variety(2, obstruction_class = 1)

        Magma input to compute radical Decomposition
        >>> s = p.to_magma()
        >>> print(s.split('ring and ideal')[1].strip())    #doctest: +ELLIPSIS  +NORMALIZE_WHITESPACE
        R<c_0011_0, c_0101_0> := PolynomialRing(RationalField(), 2, "grevlex");
        MyIdeal := ideal<R | - c_0011_0 * c_0101_0 + c_0011_0^2 + c_0101_0^2,
        ...
        >>> "RadicalDecomposition" in p.to_magma()
        True
        """
        if os.path.isfile(template_path):
            template = open(template_path, 'r').read()
        else:
            from snappy.ptolemy import __path__ as base_paths
            abs_path = os.path.join(base_paths[0], template_path)
            if os.path.isfile(abs_path):
                template = open(abs_path, 'r').read()
            else:
                raise Exception('No file at template_path %s' % template_path)
        PREAMBLE = '==TRIANGULATION=BEGINS==\n' + self._manifold._to_string() + '\n==TRIANGULATION=ENDS==\n' + 'PY=EVAL=SECTION=BEGINS=HERE\n' + self.py_eval_section() + '\nPY=EVAL=SECTION=ENDS=HERE\n'
        QUOTED_PREAMBLE = utilities.quote_ascii_text(utilities.break_long_lines(PREAMBLE))
        return Template(template).safe_substitute(PREAMBLE=PREAMBLE, QUOTED_PREAMBLE=QUOTED_PREAMBLE, VARIABLES=', '.join(self.variables), VARIABLES_QUOTED=', '.join(['"%s"' % v for v in self.variables]), VARIABLE_NUMBER=len(self.variables), VARIABLES_WITH_NON_ZERO_CONDITION=', '.join(self.variables_with_non_zero_condition), VARIABLES_WITH_NON_ZERO_CONDITION_QUOTED=', '.join(['"%s"' % v for v in self.variables_with_non_zero_condition]), VARIABLE_WITH_NON_ZERO_CONDITION_NUMBER=len(self.variables_with_non_zero_condition), EQUATIONS=',\n          '.join([str(eqn) for eqn in self.equations]), EQUATIONS_WITH_NON_ZERO_CONDITION=',\n          '.join([str(eqn) for eqn in self.equations_with_non_zero_condition]))

    def filename_base(self):
        """
        Preferred filename base for writing out this Ptolemy variety

        >>> from snappy import *
        >>> M = Manifold('4_1')
        >>> p = M.ptolemy_variety(2, obstruction_class = 1)
        >>> p.filename_base()
        '4_1__sl2_c1'

        >>> p = M.ptolemy_variety(2)
        >>> p.filename_base()
        '4_1__sl2_c0'
        """
        obstruction_class = 'NoIndex'
        if self._obstruction_class is None:
            obstruction_class = '0'
        elif self._obstruction_class._index is not None:
            obstruction_class = '%d' % self._obstruction_class._index
        name = self._manifold.name().replace('[', '_').replace(']', '')
        return '%s__sl%d_c%s' % (name, self._N, obstruction_class)

    def path_to_file(self):
        name = self._manifold.name()
        if re.match('([msvt]|o9_)[0-9]+$', name):
            dir = 'OrientableCuspedCensus'
        elif re.match('[0-9]+([\\^][0-9]+)?[_][0-9]+$', name):
            dir = 'LinkExteriors'
        elif re.match('[KL][0-9]+[an][0-9]+$', name):
            dir = 'HTLinkExteriors'
        else:
            raise Exception('No canonical path for manifold')
        tets = self._manifold.num_tetrahedra()
        return '/'.join(['data', 'pgl%d' % self._N, dir, '%02d_tetrahedra' % tets])

    def _solution_file_url(self, data_url=None, rur=False):
        if data_url is None:
            from . import DATA_URL as data_url
        if '://' not in data_url:
            if not data_url[0] == '/':
                data_url = '/' + data_url
            data_url = 'file://' + data_url
        if not data_url[-1] == '/':
            data_url = data_url + '/'
        if rur:
            ext = '.rur'
        else:
            ext = '.magma_out'
        filename = self.filename_base() + ext
        return data_url + self.path_to_file() + '/' + urlquote(filename)

    def _retrieve_solution_file(self, data_url=None, prefer_rur=False, verbose=False):
        url = self._solution_file_url(data_url=data_url, rur=prefer_rur)
        if verbose:
            print('Trying to retrieve solutions from %s ...' % url)
        try:
            return _retrieve_url(url)
        except PtolemyFileMissingError:
            url = self._solution_file_url(data_url=data_url, rur=not prefer_rur)
            if verbose:
                print('Retrieving solutions instead from %s ...:' % url)
            return _retrieve_url(url)

    def retrieve_decomposition(self, data_url=None, verbose=True):
        url = self._solution_file_url(data_url=data_url, rur=False)
        if verbose:
            print('Retrieving decomposition from %s ...' % url)
        text = _retrieve_url(url)
        if verbose:
            print('Parsing...')
        M = processFileBase.get_manifold(text)
        assert M._to_bytes() == self._manifold._to_bytes(), 'Manifold does not match census manifold'
        return processMagmaFile.decomposition_from_magma(text)

    def retrieve_solutions(self, numerical=False, prefer_rur=False, data_url=None, verbose=True):
        text = self._retrieve_solution_file(data_url=data_url, prefer_rur=prefer_rur, verbose=verbose)
        if verbose:
            print('Parsing...')
        M = processFileBase.get_manifold(text)
        assert M._to_bytes() == self._manifold._to_bytes(), 'Manifold does not match census manifold'
        return processFileDispatch.parse_solutions(text, numerical=numerical)

    def __repr__(self):
        res = 'Ptolemy Variety for %s, N = %d' % (self._manifold.name(), self._N)
        if self._obstruction_class is not None:
            res += ', obstruction_class = '
            if self._obstruction_class._index is not None:
                res += '%d' % self._obstruction_class._index
                if isinstance(self._obstruction_class, PtolemyGeneralizedObstructionClass):
                    res += ' (generalized)'
            elif isinstance(self._obstruction_class, PtolemyGeneralizedObstructionClass):
                res += '%s' % self._obstruction_class.H2_class
            else:
                res += '...'
        res += '\n' + '\n'.join(['    %s' % e for e in self.equations])
        return res

    def compute_decomposition(self, engine=None, memory_limit=750000000, directory=None, verbose=False, template_path='magma/default.magma_template'):
        """
        Starts an engine such as magma to compute the
        radical decomposition of the Ptolemy variety.

        If started in sage, uses sage, otherwise, uses magma.

        === Arguments ===

        engine --- engine to use, currently, only support magma and sage
        memory_limit --- maximal allowed memory in bytes
        directory --- location for input and output files, temporary directory used if not specified
        verbose --- print extra information
        """
        if engine is None:
            if _within_sage:
                engine = 'sage'
            else:
                engine = 'magma'
        if engine == 'magma':
            return processMagmaFile.run_magma(self.to_magma(template_path=template_path), filename_base=self.filename_base(), memory_limit=memory_limit, directory=directory, verbose=verbose)
        if engine == 'sage':
            M = self._manifold.copy()
            radical = self.ideal_with_non_zero_condition.radical()
            sage_radical_decomp = radical.primary_decomposition()

            def process_component(component):
                dimension = component.dimension()
                if dimension == 0:
                    sage_gb = component.groebner_basis()
                    polys = [Polynomial.parse_string(str(p)) for p in sage_gb]
                else:
                    polys = []
                return PtolemyVarietyPrimeIdealGroebnerBasis(polys=polys, term_order='lex', size=None, dimension=dimension, is_prime=component.is_prime(), free_variables=None, py_eval=eval(self.py_eval_section()), manifold_thunk=lambda: M)
            return utilities.MethodMappingList([process_component(component) for component in sage_radical_decomp if not component.is_one()])

    def compute_solutions(self, engine=None, numerical=False, template_path='magma/default.magma_template', memory_limit=750000000, directory=None, verbose=False):
        """
        Starts an engine such as magma to compute the
        radical decomposition of the ideal and computes exact solutions.

        If started in sage, uses sage, otherwise, uses magma.

        === Arguments ===

        engine --- engine to use, currently, only support magma and sage
        numerical --- get numerical solutions from magma instead of exact ones
        memory_limit --- maximal allowed memory in bytes
        directory --- location for input and output files, temporary directory used if not specified
        verbose --- print extra information
        """
        decomposition = self.compute_decomposition(engine=engine, memory_limit=memory_limit, template_path=template_path, directory=directory, verbose=verbose)
        return utilities.MethodMappingList([component.solutions(numerical=numerical) for component in decomposition])

    def degree_to_shapes(self):
        """
        In general, there can be d different solutions to the (reduced) Ptolemy
        variety for each solution to the gluing equations (with peripheral
        equations). This method computes d which is also the number of elements
        in H^1(Mhat; Z/N) where Mhat is the non-manifold cell comples obtained
        by gluing together the tetrahedra as non-ideal tetrahedra.


        For example, the Ptolemy variety for m009 has 4 points but there are
        only 2 distinct boundary-unipotent PSL(2,C) representations.
        Thus the following call returns 2 = 4 / 2

        >>> from snappy import Manifold
        >>> Manifold("m009").ptolemy_variety(2,1).degree_to_shapes()
        2

        >>> Manifold("m010").ptolemy_variety(2,1).degree_to_shapes()
        2
        >>> Manifold("m011").ptolemy_variety(2,1).degree_to_shapes()
        1

        >>> Manifold("m009").ptolemy_variety(3,1).degree_to_shapes()
        1
        >>> Manifold("m010").ptolemy_variety(3,1).degree_to_shapes()
        3
        >>> Manifold("m011").ptolemy_variety(3,1).degree_to_shapes()
        1

        """
        d2 = self._manifold._ptolemy_equations_boundary_map_2()[0]
        d1 = self._manifold._ptolemy_equations_boundary_map_1()[0]
        co_d1 = matrix.matrix_transpose(d2)
        co_d0 = matrix.matrix_transpose(d1)
        cohomology_classes = homology.homology_representatives(co_d1, co_d0, self._N)
        return len(list(cohomology_classes))

    def equations_as_dicts(self, with_non_zero=True):
        if with_non_zero:
            equations = self.equations_with_non_zero_condition
            variables = self.variables + ['t']
        else:
            equations = self.equations
            variables = self.variables
        result = []
        for f in equations:
            result.append({tuple((m.degree(v) for v in variables)): m.get_coefficient() for m in f.get_monomials()})
        return result