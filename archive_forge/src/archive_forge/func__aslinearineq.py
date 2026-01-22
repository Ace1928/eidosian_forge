from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
def _aslinearineq(self):
    """ 
        Converts a convex PWL inequality into an equivalent set of 
        linear inequalities. 

        Returns a tuple (ineqs, aux_ineqs, aux_vars).  

        If self is a linear inequailty, then ineqs = [self], 
        aux_ineqs = [], aux_vars = [].

        If self is PWL then ineqs and aux_ineqs are two lists of 
        linear inequalities that together are equivalent to self.
        They are separated in two sets so that the multiplier for self 
        depends only on the multipliers of the constraints in ineqs:
        - if len(self) == max(len(ineqs[k])), then the multiplier of 
          self is sum_k ineqs[k].multiplier
        - if len(self) == max(len(ineqs[k])), then the multiplier of 
          self is sum(sum_k ineqs[k].multiplier)

        aux_vars is a varlist with new auxiliary variables.
        """
    if self.type() != '<':
        raise TypeError('constraint must be an inequality')
    ineqs, aux_ineqs, aux_vars = ([], [], varlist())
    faff = _function()
    faff._constant = self._f._constant
    faff._linear = self._f._linear
    cvxterms = self._f._cvxterms
    if not cvxterms:
        ineqs += [self]
    elif len(cvxterms) == 1 and type(cvxterms[0]) is _minmax:
        if len(cvxterms[0]._flist) == 1:
            f0 = cvxterms[0]._flist[0]
            if len(faff) == 1:
                c = faff + f0 <= 0
                c.name = self.name
                c, caux, newvars = c._aslinearineq()
                ineqs += c
                aux_ineqs += caux
                aux_vars += newvars
            else:
                for k in range(len(f0)):
                    c = faff + f0[k] <= 0
                    c.name = self.name + '(%d)' % k
                    c, caux, newvars = c._aslinearineq()
                    ineqs += c
                    aux_ineqs += caux
                    aux_vars += newvars
        else:
            for k in range(len(cvxterms[0]._flist)):
                c = faff + cvxterms[0]._flist[k] <= 0
                c.name = self.name + '(%d)' % k
                c, caux, newvars = c._aslinearineq()
                ineqs += c
                aux_ineqs += caux
                aux_vars += newvars
    else:
        sumt = _function()
        for k in range(len(cvxterms)):
            if type(cvxterms[k]) is _minmax:
                tk = variable(len(cvxterms[k]), self.name + '_x' + str(k))
                aux_vars += [tk]
                sumt = sumt + tk
                if len(cvxterms[k]._flist) == 1:
                    f0 = cvxterms[k]._flist[0]
                    c = f0 <= tk
                    c.name = self.name + '[%d]' % k
                    c, caux, newvars = c._aslinearineq()
                    aux_ineqs += c + caux
                    aux_vars += newvars
                else:
                    for j in range(len(cvxterms[k]._flist)):
                        fj = cvxterms[k]._flist[j]
                        c = fj <= tk
                        c.name = self.name + '[%d](%d)' % (k, j)
                        c, caux, newvars = c._aslinearineq()
                        aux_ineqs += c + caux
                        aux_vars += newvars
            else:
                tk = variable(cvxterms[k]._length(), self.name + '_x' + str(k))
                aux_vars += [tk]
                sumt = sumt + sum(tk)
                for j in range(len(cvxterms[k]._flist)):
                    fj = cvxterms[k]._flist[j]
                    c = fj <= tk
                    c.name = self.name + '[%d](%d)' % (k, j)
                    c, caux, newvars = c._aslinearineq()
                    aux_ineqs += c + caux
                    aux_vars += newvars
        c = faff + sumt <= 0
        c.name = self.name
        ineqs += [c]
    return (ineqs, aux_ineqs, aux_vars)