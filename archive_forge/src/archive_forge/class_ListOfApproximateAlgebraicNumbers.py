from sage.all import (cached_method, real_part, imag_part, round, ceil, floor, log,
import itertools
class ListOfApproximateAlgebraicNumbers:

    def __init__(self, defining_function):
        self.f = defining_function
        self.n = len(defining_function(100))
        self._field = {True: None, False: None}

    @cached_method
    def __call__(self, prec):
        return self.f(prec)

    @cached_method
    def __getitem__(self, index):

        def f(n):
            return self(n)[index]
        return ApproximateAlgebraicNumber(f)

    def list(self):
        return [self[i] for i in range(self.n)]

    def __repr__(self):
        return '<SetOfAAN: %s>' % [CDF(z) for z in self.f(100)]

    def __len__(self):
        return self.n

    def are_in_field_generated_by(self, z, prec=None):
        ans = []
        for i in range(self.n):
            p = z.express(self[i], prec)
            if p is None:
                return None
            ans.append(p)
        return ans

    def _find_field_uncached(self, prec, degree, verbosity=False):

        def min_poly(z):
            return z.min_polynomial(prec, degree)

        def min_poly_deg(z):
            return z.min_polynomial().degree()

        def message(*args):
            if verbosity:
                print(*args)
        elts = self.list()
        z = ApproximateAlgebraicNumber(1)
        z.min_polynomial()
        exact_elts = []
        for i, elt in enumerate(elts):
            exact_elt = z.express(elt, prec)
            if exact_elt is not None:
                exact_elts.append(exact_elt)
            else:
                if min_poly(elt) is None:
                    message('Bailing: no minimal polynomial found for newly considered element', elt)
                    return None
                found = False
                candidates = [elt, z + elt, z - elt, z * elt, elt + elt + z, z + z + elt, elt + elt - z, z + z - elt, z + z * elt, elt + elt * z]
                for w in candidates:
                    if min_poly(w) is None:
                        message('Skipping: no minimal polynomial found for possible primitive element', elt)
                    elif min_poly_deg(w) >= min_poly_deg(z) and w.can_express(z, prec) and w.can_express(elt, prec):
                        exact_elts = w.express_several(elts[:i + 1], prec)
                        if exact_elts is None:
                            message("Bailing: Couldn't express everythingin terms of primitive element")
                            return None
                        found, z = (True, w)
                        break
                if not found:
                    message("Bailing: Couldn't find primitive element for larger field")
                    return None
        field = z.number_field()
        exact_elts = [field(exact_elt) for exact_elt in exact_elts]
        return (field, z, exact_elts)

    def find_field(self, prec, degree, optimize=False, verbosity=False):
        if self._field[False] is None:
            self._field[False] = self._find_field_uncached(prec, degree, verbosity)
        if optimize and self._field[True] is None:
            if self._field[False] is None:
                return None
            K, z, exact_elts = self._field[False]
            z = optimize_field_generator(z)
            exact_elts = self.are_in_field_generated_by(z, prec)
            if exact_elts is None:
                if verbosity:
                    print('Bailing: Could not express elements in optimized generator', z)
                return None
            field = z.number_field()
            exact_elts = [field(exact_elt) for exact_elt in exact_elts]
            self._field[True] = (field, z, exact_elts)
        return self._field[optimize]

    def _as_exact_matrices(self, optimize=None):
        if optimize is None:
            optimize = self._field[True] is not None
        if len(self) % 4:
            raise ValueError('Not right number of values to form 2x2 matrices')
        K, z, ans = self._field[optimize]
        return (z, [matrix(K, 2, 2, ans[n:n + 4]) for n in range(0, len(ans), 4)])