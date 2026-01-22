from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
class op(object):
    """
    An optimization problem.

    op(objective=0.0, constraints=None, name '') constructs an 
    optimization problem.


    Arguments:

    objective       scalar (int, float, 1x1 dense 'd' matrix), scalar 
                    variable, scalar affine function or scalar convex
                    piecewise-linear function.  Scalars and variables 
                    are converted to affine functions.  
    constraints     None, a single constraint, or a list of constraints
                    None means the same as an empty list.  A single 
                    constraint means the same as a singleton list.
    name            string with the name of the LP


    Attributes:

    objective       the objective function (borrowed reference to the 
                    function passed as argument).  
    name            the name of the optimization problem
    status          initially None.  After solving the problem, 
                    summarizes the outcome.
    _inequalities   list of inequality constraints 
    _equalities     list of equality constraints 
    _variables      a dictionary {v: dictionary with keys 'o','i','e'}
                    The keys v are the variables in the problem.
                    'o': True/False depending on whether v appears in 
                    the objective or not;
                    'i': list of inequality constraints v appears in;
                    'e': list of equality constraints v appears in.
               

    Methods:

    variables()     returns a list of variables.  The list is a varlist
                    (defined below), ie, a subclass of 'list'.
    constraints()   returns a list of constraints 
    inequalities()  returns a list of inequality constraints
    equalities()    returns a list of equality constraints
    delconstraint() deletes a constraint 
    addconstraint() adds a constraint
    _inmatrixform() returns an equivalent LP in matrix form
    solve()         solves the problem
    tofile()        if the problem is an LP, writes it to an MPS file
    fromfile()      reads an LP from an MPS file
    """

    def __init__(self, objective=0.0, constraints=None, name=''):
        self._variables = dict()
        self.objective = objective
        for v in self.objective.variables():
            self._variables[v] = {'o': True, 'i': [], 'e': []}
        self._inequalities, self._equalities = ([], [])
        if constraints is None:
            pass
        elif type(constraints) is constraint:
            if constraints.type() == '<':
                self._inequalities += [constraints]
            else:
                self._equalities += [constraints]
        elif type(constraints) == list and (not [c for c in constraints if type(c) is not constraint]):
            for c in constraints:
                if c.type() == '<':
                    self._inequalities += [c]
                else:
                    self._equalities += [c]
        else:
            raise TypeError('invalid argument for constraints')
        for c in self._inequalities:
            for v in c.variables():
                if v in self._variables:
                    self._variables[v]['i'] += [c]
                else:
                    self._variables[v] = {'o': False, 'i': [c], 'e': []}
        for c in self._equalities:
            for v in c.variables():
                if v in self._variables:
                    self._variables[v]['e'] += [c]
                else:
                    self._variables[v] = {'o': False, 'i': [], 'e': [c]}
        self.name = name
        self.status = None

    def __repr__(self):
        n = sum(map(len, self._variables))
        m = sum(map(len, self._inequalities))
        p = sum(map(len, self._equalities))
        return '<optimization problem with %d variables, %d inequality and %d equality constraint(s)>' % (n, m, p)

    def __setattr__(self, name, value):
        if name == 'objective':
            if _isscalar(value):
                value = _function() + value
            elif type(value) is variable and len(value) == 1:
                value = +value
            elif type(value) is _function and value._isconvex() and (len(value) == 1):
                pass
            else:
                raise TypeError("attribute 'objective' must be a scalar affine or convex PWL function")
            for v in self.variables():
                if not self._variables[v]['i'] and (not self._variables[v]['e']):
                    del self._variables[v]
            object.__setattr__(self, 'objective', value)
            for v in self.objective.variables():
                if v not in self._variables:
                    self._variables[v] = {'o': True, 'i': [], 'e': []}
                else:
                    self._variables[v]['o'] = True
        elif name == 'name':
            if type(value) is str:
                object.__setattr__(self, name, value)
            else:
                raise TypeError("attribute 'name' must be string")
        elif name == '_inequalities' or name == '_equalities' or name == '_variables' or (name == 'status'):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError("'op' object has no attribute '%s'" % name)

    def variables(self):
        """ Returns a list of variables of the LP. """
        return varlist(self._variables.keys())

    def constraints(self):
        """ Returns a list of constraints of the LP."""
        return self._inequalities + self._equalities

    def equalities(self):
        """ Returns a list of equality constraints of the LP."""
        return list(self._equalities)

    def inequalities(self):
        """ Returns a list of inequality constraints of the LP."""
        return list(self._inequalities)

    def delconstraint(self, c):
        """ 
        Deletes constraint c from the list of constrains  
        """
        if type(c) is not constraint:
            raise TypeError("argument must be of type 'constraint'")
        try:
            if c.type() == '<':
                self._inequalities.remove(c)
                for v in c.variables():
                    self._variables[v]['i'].remove(c)
            else:
                self._equalities.remove(c)
                for v in c.variables():
                    self._variables[v]['e'].remove(c)
            if not self._variables[v]['o'] and (not self._variables[v]['i']) and (not self._variables[v]['e']):
                del self._variables[v]
        except ValueError:
            pass

    def addconstraint(self, c):
        """ 
        Adds constraint c to the list of constraints. 
        """
        if type(c) is not constraint:
            raise TypeError('argument must be of type constraint')
        if c.type() == '<':
            self._inequalities += [c]
        if c.type() == '=':
            self._equalities += [c]
        for v in c.variables():
            if c.type() == '<':
                if v in self._variables:
                    self._variables[v]['i'] += [c]
                else:
                    self._variables[v] = {'o': False, 'i': [c], 'e': []}
            elif v in self._variables:
                self._variables[v]['e'] += [c]
            else:
                self._variables[v] = {'o': False, 'i': [], 'e': [c]}

    def _islp(self):
        """ 
        Returns True if self is an LP; False otherwise.
        """
        if not self.objective._isaffine():
            return False
        for c in self._inequalities:
            if not c._f._isaffine():
                return False
        for c in self._equalities:
            if not c._f._isaffine():
                return False
        return True

    def _inmatrixform(self, format='dense'):
        """ 
        Converts self to an LP in matrix form 

                minimize    c'*x+d
                subject to  G*x <= h
                            A*x = b.

        c, h, b are dense column matrices; G and A sparse or dense 
        matrices depending on format ('sparse' or 'dense').   

        If self is already an LP in matrix form with the correct matrix
        types, then _inmatrixform() returns None.  Otherwise it returns 
        a tuple (newlp, vmap, mmap).

        newlp is an LP in matrix form with the correct format and 
        matrix types.

        vmap is a dictionary with the variables of self as keys and
        affine functions as values.  For each variable v of self, 
        vmap[v] is a function of the new variable x that can be 
        evaluated to obtain the solution v from the solution x.

        mmap is a dictionary with the constraints of self as keys and
        affine functions as values.  For each constraint c of self, 
        mmap[c] is a function of the multipliers of the new lp that can
        be evaluated to obtain the optimal multiplier for c.
        """
        variables, aux_variables = (self.variables(), varlist())
        lin_ineqs, pwl_ineqs, aux_ineqs = ([], dict(), [])
        for i in self._inequalities:
            if i._f._isaffine():
                lin_ineqs += [i]
            else:
                pwl_ineqs[i] = []
        equalities = self._equalities
        objective = +self.objective
        if objective._isaffine() and len(variables) == 1 and (not pwl_ineqs) and (len(lin_ineqs) <= 1) and (len(equalities) <= 1):
            v = variables[0]
            if lin_ineqs:
                G = lin_ineqs[0]._f._linear._coeff[v]
            else:
                G = None
            if equalities:
                A = equalities[0]._f._linear._coeff[v]
            else:
                A = None
            if format == 'dense' and (G is None or _isdmatrix(G)) and (A is None or _isdmatrix(A)) or (format == 'sparse' and (G is None or _isspmatrix(G)) and (A is None or _isspmatrix(A))):
                return None
        if not objective._isaffine():
            newobj = _function()
            newobj._constant = +objective._constant
            newobj._linear = +objective._linear
            for k in range(len(objective._cvxterms)):
                fk = objective._cvxterms[k]
                if type(fk) is _minmax:
                    tk = variable(1, self.name + '_x' + str(k))
                    newobj += tk
                else:
                    tk = variable(fk._length(), self.name + '_x' + str(k))
                    newobj += sum(tk)
                aux_variables += [tk]
                for j in range(len(fk._flist)):
                    c = fk._flist[j] <= tk
                    if len(fk._flist) > 1:
                        c.name = self.name + '[%d](%d)' % (k, j)
                    else:
                        c.name = self.name + '[%d]' % k
                    c, caux, newvars = c._aslinearineq()
                    aux_ineqs += c + caux
                    aux_variables += newvars
            objective = newobj
        for i in pwl_ineqs:
            pwl_ineqs[i], caux, newvars = i._aslinearineq()
            aux_ineqs += caux
            aux_variables += newvars
        vslc = dict()
        n = 0
        for v in variables + aux_variables:
            vslc[v] = slice(n, n + len(v))
            n += len(v)
        c = matrix(0.0, (1, n))
        for v, cf in iter(objective._linear._coeff.items()):
            if _isscalar(cf):
                c[vslc[v]] = cf[0]
            elif _isdmatrix(cf):
                c[vslc[v]] = cf[:]
            else:
                c[vslc[v]] = matrix(cf[:], tc='d')
        if n > 0:
            x = variable(n)
            cost = c * x + objective._constant
        else:
            cost = _function() + objective._constant[0]
        vmap = dict()
        for v in variables:
            vmap[v] = x[vslc[v]]
        islc = dict()
        for i in lin_ineqs + aux_ineqs:
            islc[i] = None
        for c in pwl_ineqs:
            for i in pwl_ineqs[c]:
                islc[i] = None
        m = 0
        for i in islc:
            islc[i] = slice(m, m + len(i))
            m += len(i)
        if format == 'sparse':
            G = spmatrix(0.0, [], [], (m, n))
        else:
            G = matrix(0.0, (m, n))
        h = matrix(0.0, (m, 1))
        for i in islc:
            lg = len(i)
            for v, cf in iter(i._f._linear._coeff.items()):
                if cf.size == (lg, len(v)):
                    if _isspmatrix(cf) and _isdmatrix(G):
                        G[islc[i], vslc[v]] = matrix(cf, tc='d')
                    else:
                        G[islc[i], vslc[v]] = cf
                elif cf.size == (1, len(v)):
                    if _isspmatrix(cf) and _isdmatrix(G):
                        G[islc[i], vslc[v]] = matrix(cf[lg * [0], :], tc='d')
                    else:
                        G[islc[i], vslc[v]] = cf[lg * [0], :]
                else:
                    G[islc[i].start + m * vslc[v].start:islc[i].stop + m * vslc[v].stop:m + 1] = cf[0]
            if _isscalar(i._f._constant):
                h[islc[i]] = -i._f._constant[0]
            else:
                h[islc[i]] = -i._f._constant[:]
        eslc = dict()
        p = 0
        for e in equalities:
            eslc[e] = slice(p, p + len(e))
            p += len(e)
        if format == 'sparse':
            A = spmatrix(0.0, [], [], (p, n))
        else:
            A = matrix(0.0, (p, n))
        b = matrix(0.0, (p, 1))
        for e in equalities:
            lg = len(e)
            for v, cf in iter(e._f._linear._coeff.items()):
                if cf.size == (lg, len(v)):
                    if _isspmatrix(cf) and _isdmatrix(A):
                        A[eslc[e], vslc[v]] = matrix(cf, tc='d')
                    else:
                        A[eslc[e], vslc[v]] = cf
                elif cf.size == (1, len(v)):
                    if _isspmatrix(cf) and _isdmatrix(A):
                        A[eslc[e], vslc[v]] = matrix(cf[lg * [0], :], tc='d')
                    else:
                        A[eslc[e], vslc[v]] = cf[lg * [0], :]
                else:
                    A[eslc[e].start + p * vslc[v].start:eslc[e].stop + p * vslc[v].stop:p + 1] = cf[0]
            if _isscalar(e._f._constant):
                b[eslc[e]] = -e._f._constant[0]
            else:
                b[eslc[e]] = -e._f._constant[:]
        constraints = []
        if n:
            if m:
                constraints += [G * x <= h]
            if p:
                constraints += [A * x == b]
        else:
            if m:
                constraints += [_function() - h <= 0]
            if p:
                constraints += [_function() - b == 0]
        mmap = dict()
        for i in lin_ineqs:
            mmap[i] = constraints[0].multiplier[islc[i]]
        for i in pwl_ineqs:
            mmap[i] = _function()
            for c in pwl_ineqs[i]:
                mmap[i] = mmap[i] + constraints[0].multiplier[islc[c]]
            if len(i) == 1 != len(mmap[i]):
                mmap[i] = sum(mmap[i])
        for e in equalities:
            mmap[e] = constraints[1].multiplier[eslc[e]]
        return (op(cost, constraints), vmap, mmap)

    def solve(self, format='dense', solver='default', **kwargs):
        """
        Solves LP using dense or sparse solver.

        format is 'dense' or 'sparse' 

        solver is 'default', 'glpk' or 'mosek'

        solve() sets self.status, and if status is 'optimal', also 
        the value attributes of the variables and the constraint 
        multipliers.  If solver is 'python' then if status is 
        'primal infeasible', the constraint multipliers are set to
        a proof of infeasibility; if status is 'dual infeasible' the
        variables are set to a proof of dual infeasibility.
        """
        t = self._inmatrixform(format)
        if t is None:
            lp1 = self
        else:
            lp1, vmap, mmap = (t[0], t[1], t[2])
        variables = lp1.variables()
        if not variables:
            raise TypeError('lp must have at least one variable')
        x = variables[0]
        c = lp1.objective._linear._coeff[x]
        if _isspmatrix(c):
            c = matrix(c, tc='d')
        inequalities = lp1._inequalities
        if not inequalities:
            raise TypeError('lp must have at least one inequality')
        G = inequalities[0]._f._linear._coeff[x]
        h = -inequalities[0]._f._constant
        equalities = lp1._equalities
        if equalities:
            A = equalities[0]._f._linear._coeff[x]
            b = -equalities[0]._f._constant
        elif format == 'dense':
            A = matrix(0.0, (0, len(x)))
            b = matrix(0.0, (0, 1))
        else:
            A = spmatrix(0.0, [], [], (0, len(x)))
            b = matrix(0.0, (0, 1))
        sol = solvers.lp(c[:], G, h, A, b, solver=solver, **kwargs)
        x.value = sol['x']
        inequalities[0].multiplier.value = sol['z']
        if equalities:
            equalities[0].multiplier.value = sol['y']
        self.status = sol['status']
        if type(t) is tuple:
            for v, f in iter(vmap.items()):
                v.value = f.value()
            for c, f in iter(mmap.items()):
                c.multiplier.value = f.value()

    def tofile(self, filename):
        """ 
        writes LP to file 'filename' in MPS format.
        """
        if not self._islp():
            raise TypeError('problem must be an LP')
        constraints = self.constraints()
        variables = self.variables()
        inequalities = self.inequalities()
        equalities = self.equalities()
        f = open(filename, 'w')
        f.write('NAME')
        if self.name:
            f.write(10 * ' ' + self.name[:8].rjust(8))
        f.write('\n')
        f.write('ROWS\n')
        f.write(' N  %8s\n' % 'cost')
        for k in range(len(constraints)):
            c = constraints[k]
            for i in range(len(c)):
                if c._type == '<':
                    f.write(' L  ')
                else:
                    f.write(' E  ')
                if c.name:
                    name = c.name
                else:
                    name = str(k)
                name = name[:7 - len(str(i))] + '_' + str(i)
                f.write(name.rjust(8))
                f.write('\n')
        f.write('COLUMNS\n')
        for k in range(len(variables)):
            v = variables[k]
            for i in range(len(v)):
                if v.name:
                    varname = v.name
                else:
                    varname = str(k)
                varname = varname[:7 - len(str(i))] + '_' + str(i)
                if v in self.objective._linear._coeff:
                    cf = self.objective._linear._coeff[v]
                    if cf[i] != 0.0:
                        f.write(4 * ' ' + varname[:8].rjust(8))
                        f.write(2 * ' ' + '%8s' % 'cost')
                        f.write(2 * ' ' + '% 7.5E\n' % cf[i])
                for j in range(len(constraints)):
                    c = constraints[j]
                    if c.name:
                        cname = c.name
                    else:
                        cname = str(j)
                    if v in c._f._linear._coeff:
                        cf = c._f._linear._coeff[v]
                        if cf.size == (len(c), len(v)):
                            nz = [k for k in range(cf.size[0]) if cf[k, i] != 0.0]
                            for l in nz:
                                conname = cname[:7 - len(str(l))] + '_' + str(l)
                                f.write(4 * ' ' + varname[:8].rjust(8))
                                f.write(2 * ' ' + conname[:8].rjust(8))
                                f.write(2 * ' ' + '% 7.5E\n' % cf[l, i])
                        elif cf.size == (1, len(v)):
                            if cf[0, i] != 0.0:
                                for l in range(len(c)):
                                    conname = cname[:7 - len(str(l))] + '_' + str(l)
                                    f.write(4 * ' ' + varname[:8].rjust(8))
                                    f.write(2 * ' ' + conname[:8].rjust(8))
                                    f.write(2 * ' ' + '% 7.5E\n' % cf[0, i])
                        elif _isscalar(cf):
                            if cf[0, 0] != 0.0:
                                conname = cname[:7 - len(str(i))] + '_' + str(i)
                                f.write(4 * ' ' + varname[:8].rjust(8))
                                f.write(2 * ' ' + conname[:8].rjust(8))
                                f.write(2 * ' ' + '% 7.5E\n' % cf[0, 0])
        f.write('RHS\n')
        for j in range(len(constraints)):
            c = constraints[j]
            if c.name:
                cname = c.name
            else:
                cname = str(j)
            const = -c._f._constant
            for l in range(len(c)):
                conname = cname[:7 - len(str(l))] + '_' + str(l)
                f.write(14 * ' ' + conname[:8].rjust(8))
                if const.size[0] == len(c):
                    f.write(2 * ' ' + '% 7.5E\n' % const[l])
                else:
                    f.write(2 * ' ' + '% 7.5E\n' % const[0])
        f.write('RANGES\n')
        f.write('BOUNDS\n')
        for k in range(len(variables)):
            v = variables[k]
            for i in range(len(v)):
                if v.name:
                    varname = v.name
                else:
                    varname = str(k)
                varname = varname[:7 - len(str(i))] + '_' + str(i)
                f.write(' FR ' + 10 * ' ' + varname[:8].rjust(8) + '\n')
        f.write('ENDATA\n')
        f.close()

    def fromfile(self, filename):
        """ 
        Reads LP from file 'filename' assuming it is a fixed format 
        ascii MPS file.

        Does not include serious error checking. 

        MPS features that are not allowed: comments preceded by 
        dollar signs, linear combinations of rows, multiple righthand
        sides, ranges columns or bounds columns.
        """
        self._inequalities = []
        self._equalities = []
        self.objective = _function()
        self.name = ''
        f = open(filename, 'r')
        s = f.readline()
        while s[:4] != 'NAME':
            s = f.readline()
            if not s:
                raise SyntaxError("EOF reached before 'NAME' section was found")
        self.name = s[14:22].strip()
        s = f.readline()
        while s[:4] != 'ROWS':
            if not s:
                raise SyntaxError("EOF reached before 'ROWS' section was found")
            s = f.readline()
        s = f.readline()
        functions = dict()
        rowtypes = dict()
        foundobj = False
        while s[:7] != 'COLUMNS':
            if not s:
                raise SyntaxError("file has no 'COLUMNS' section")
            if len(s.strip()) == 0 or s[0] == '*':
                pass
            elif s[1:3].strip() in ['E', 'L', 'G']:
                rowlabel = s[4:12].strip()
                functions[rowlabel] = _function()
                rowtypes[rowlabel] = s[1:3].strip()
            elif s[1:3].strip() == 'N':
                rowlabel = s[4:12].strip()
                if not foundobj:
                    functions[rowlabel] = self.objective
                    foundobj = True
            else:
                raise ValueError("unknown row type '%s'" % s[1:3].strip())
            s = f.readline()
        s = f.readline()
        variables = dict()
        while s[:3] != 'RHS':
            if not s:
                raise SyntaxError("EOF reached before 'RHS' section was found")
            if len(s.strip()) == 0 or s[0] == '*':
                pass
            else:
                if s[4:12].strip():
                    collabel = s[4:12].strip()
                if collabel not in variables:
                    variables[collabel] = variable(1, collabel)
                v = variables[collabel]
                rowlabel = s[14:22].strip()
                if rowlabel not in functions:
                    raise KeyError("no row label '%s'" % rowlabel)
                functions[rowlabel]._linear._coeff[v] = matrix(float(s[24:36]), tc='d')
                rowlabel = s[39:47].strip()
                if rowlabel:
                    if rowlabel not in functions:
                        raise KeyError("no row label '%s'" % rowlabel)
                    functions[rowlabel]._linear._coeff[v] = matrix(float(s[49:61]), tc='d')
            s = f.readline()
        s = f.readline()
        rhslabel = None
        while s[:6] != 'RANGES' and s[:6] != 'BOUNDS' and (s[:6] != 'ENDATA'):
            if not s:
                raise SyntaxError("EOF reached before 'ENDATA' was found")
            if len(s.strip()) == 0 or s[0] == '*':
                pass
            elif None != rhslabel != s[4:12].strip():
                pass
            else:
                if rhslabel is None:
                    rhslabel = s[4:12].strip()
                rowlabel = s[14:22].strip()
                if rowlabel not in functions:
                    raise KeyError("no row label '%s'" % rowlabel)
                functions[rowlabel]._constant = matrix(-float(s[24:36]), tc='d')
                rowlabel = s[39:47].strip()
                if rowlabel:
                    if rowlabel not in functions:
                        raise KeyError("no row label '%s'" % rowlabel)
                    functions[rowlabel]._constant = matrix(-float(s[49:61]), tc='d')
            s = f.readline()
        ranges = dict()
        for l in iter(rowtypes.keys()):
            ranges[l] = None
        rangeslabel = None
        if s[:6] == 'RANGES':
            s = f.readline()
            while s[:6] != 'BOUNDS' and s[:6] != 'ENDATA':
                if not s:
                    raise SyntaxError("EOF reached before 'ENDATA' was found")
                if len(s.strip()) == 0 or s[0] == '*':
                    pass
                elif None != rangeslabel != s[4:12].strip():
                    pass
                else:
                    if rangeslabel == None:
                        rangeslabel = s[4:12].strip()
                    rowlabel = s[14:22].strip()
                    if rowlabel not in rowtypes:
                        raise KeyError("no row label '%s'" % rowlabel)
                    ranges[rowlabel] = float(s[24:36])
                    rowlabel = s[39:47].strip()
                    if rowlabel != '':
                        if rowlabel not in functions:
                            raise KeyError("no row label '%s'" % rowlabel)
                        ranges[rowlabel] = float(s[49:61])
                s = f.readline()
        boundslabel = None
        bounds = dict()
        for v in iter(variables.keys()):
            bounds[v] = [0.0, None]
        if s[:6] == 'BOUNDS':
            s = f.readline()
            while s[:6] != 'ENDATA':
                if not s:
                    raise SyntaxError("EOF reached before 'ENDATA' was found")
                if len(s.strip()) == 0 or s[0] == '*':
                    pass
                elif None != boundslabel != s[4:12].strip():
                    pass
                else:
                    if boundslabel is None:
                        boundslabel = s[4:12].strip()
                    collabel = s[14:22].strip()
                    if collabel not in variables:
                        raise ValueError('unknown column label ' + "'%s'" % collabel)
                    if s[1:3].strip() == 'LO':
                        if bounds[collabel][0] != 0.0:
                            raise ValueError("repeated lower bound for variable '%s'" % collabel)
                        bounds[collabel][0] = float(s[24:36])
                    elif s[1:3].strip() == 'UP':
                        if bounds[collabel][1] != None:
                            raise ValueError("repeated upper bound for variable '%s'" % collabel)
                        bounds[collabel][1] = float(s[24:36])
                    elif s[1:3].strip() == 'FX':
                        if bounds[collabel] != [0, None]:
                            raise ValueError("repeated bounds for variable '%s'" % collabel)
                        bounds[collabel][0] = float(s[24:36])
                        bounds[collabel][1] = float(s[24:36])
                    elif s[1:3].strip() == 'FR':
                        if bounds[collabel] != [0, None]:
                            raise ValueError("repeated bounds for variable '%s'" % collabel)
                        bounds[collabel][0] = None
                        bounds[collabel][1] = None
                    elif s[1:3].strip() == 'MI':
                        if bounds[collabel][0] != 0.0:
                            raise ValueError("repeated lower bound for variable '%s'" % collabel)
                        bounds[collabel][0] = None
                    elif s[1:3].strip() == 'PL':
                        if bounds[collabel][1] != None:
                            raise ValueError("repeated upper bound for variable '%s'" % collabel)
                    else:
                        raise ValueError("unknown bound type '%s'" % s[1:3].strip())
                s = f.readline()
        for l, type in iter(rowtypes.items()):
            if type == 'L':
                c = functions[l] <= 0.0
                c.name = l
                self._inequalities += [c]
                if ranges[l] != None:
                    c = functions[l] >= -abs(ranges[l])
                    c.name = l + '_lb'
                    self._inequalities += [c]
            if type == 'G':
                c = functions[l] >= 0.0
                c.name = l
                self._inequalities += [c]
                if ranges[l] != None:
                    c = functions[l] <= abs(ranges[l])
                    c.name = l + '_ub'
                    self._inequalities += [c]
            if type == 'E':
                if ranges[l] is None or ranges[l] == 0.0:
                    c = functions[l] == 0.0
                    c.name = l
                    self._equalities += [c]
                elif ranges[l] > 0.0:
                    c = functions[l] >= 0.0
                    c.name = l + '_lb'
                    self._inequalities += [c]
                    c = functions[l] <= ranges[l]
                    c.name = l + '_ub'
                    self._inequalities += [c]
                else:
                    c = functions[l] <= 0.0
                    c.name = l + '_ub'
                    self._inequalities += [c]
                    c = functions[l] >= ranges[l]
                    c.name = l + '_lb'
                    self._inequalities += [c]
        for l, bnds in iter(bounds.items()):
            v = variables[l]
            if None != bnds[0] != bnds[1]:
                c = v >= bnds[0]
                self._inequalities += [c]
            if bnds[0] != bnds[1] != None:
                c = v <= bnds[1]
                self._inequalities += [c]
            if None != bnds[0] == bnds[1]:
                c = v == bnds[0]
                self._equalities += [c]
        for c in self._inequalities + self._equalities:
            if len(c._f._linear._coeff) == 0:
                if c.type() == '=' and c._f._constant[0] != 0.0:
                    raise ValueError("equality constraint '%s' has no variables and a nonzero righthand side" % c.name)
                elif c.type() == '<' and c._f._constant[0] > 0.0:
                    raise ValueError("inequality constraint '%s' has no variables and a negative righthand side" % c.name)
                else:
                    print("removing redundant constraint '%s'" % c.name)
                    if c.type() == '<':
                        self._inequalities.remove(c)
                    if c.type() == '=':
                        self._equalities.remove(c)
        self._variables = dict()
        for v in self.objective._linear._coeff.keys():
            self._variables[v] = {'o': True, 'i': [], 'e': []}
        for c in self._inequalities:
            for v in c._f._linear._coeff.keys():
                if v in self._variables:
                    self._variables[v]['i'] += [c]
                else:
                    self._variables[v] = {'o': False, 'i': [c], 'e': []}
        for c in self._equalities:
            for v in c._f._linear._coeff.keys():
                if v in self._variables:
                    self._variables[v]['e'] += [c]
                else:
                    self._variables[v] = {'o': False, 'i': [], 'e': [c]}
        self.status = None
        f.close()