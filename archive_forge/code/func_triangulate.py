import copy
import logging
import itertools
import decimal
from functools import cache
import numpy
from ._vertex import (VertexCacheField, VertexCacheIndex)
def triangulate(self, n=None, symmetry=None, centroid=True, printout=False):
    """
        Triangulate the initial domain, if n is not None then a limited number
        of points will be generated

        Parameters
        ----------
        n : int, Number of points to be sampled.
        symmetry :

            Ex. Dictionary/hashtable
            f(x) = (x_1 + x_2 + x_3) + (x_4)**2 + (x_5)**2 + (x_6)**2

            symmetry = symmetry[0]: 0,  # Variable 1
                       symmetry[1]: 0,  # symmetric to variable 1
                       symmetry[2]: 0,  # symmetric to variable 1
                       symmetry[3]: 3,  # Variable 4
                       symmetry[4]: 3,  # symmetric to variable 4
                       symmetry[5]: 3,  # symmetric to variable 4
                        }
        centroid : bool, if True add a central point to the hypercube
        printout : bool, if True print out results

        NOTES:
        ------
        Rather than using the combinatorial algorithm to connect vertices we
        make the following observation:

        The bound pairs are similar a C2 cyclic group and the structure is
        formed using the cartesian product:

        H = C2 x C2 x C2 ... x C2 (dim times)

        So construct any normal subgroup N and consider H/N first, we connect
        all vertices within N (ex. N is C2 (the first dimension), then we move
        to a left coset aN (an operation moving around the defined H/N group by
        for example moving from the lower bound in C2 (dimension 2) to the
        higher bound in C2. During this operation connection all the vertices.
        Now repeat the N connections. Note that these elements can be connected
        in parallel.
        """
    if symmetry is None:
        symmetry = self.symmetry
    origin = [i[0] for i in self.bounds]
    self.origin = origin
    supremum = [i[1] for i in self.bounds]
    self.supremum = supremum
    if symmetry is None:
        cbounds = self.bounds
    else:
        cbounds = copy.copy(self.bounds)
        for i, j in enumerate(symmetry):
            if i is not j:
                cbounds[i] = [self.bounds[symmetry[i]][0]]
                cbounds[i] = [self.bounds[symmetry[i]][1]]
                if self.bounds[symmetry[i]] is not self.bounds[symmetry[j]]:
                    logging.warning(f'Variable {i} was specified as symmetetric to variable {j}, however, the bounds {i} = {self.bounds[symmetry[i]]} and {j} = {self.bounds[symmetry[j]]} do not match, the mismatch was ignored in the initial triangulation.')
                    cbounds[i] = self.bounds[symmetry[j]]
    if n is None:
        self.cp = self.cyclic_product(cbounds, origin, supremum, centroid)
        for i in self.cp:
            i
        try:
            self.triangulated_vectors.append((tuple(self.origin), tuple(self.supremum)))
        except (AttributeError, KeyError):
            self.triangulated_vectors = [(tuple(self.origin), tuple(self.supremum))]
    else:
        try:
            self.cp
        except (AttributeError, KeyError):
            self.cp = self.cyclic_product(cbounds, origin, supremum, centroid)
        try:
            while len(self.V.cache) < n:
                next(self.cp)
        except StopIteration:
            try:
                self.triangulated_vectors.append((tuple(self.origin), tuple(self.supremum)))
            except (AttributeError, KeyError):
                self.triangulated_vectors = [(tuple(self.origin), tuple(self.supremum))]
    if printout:
        for v in self.V.cache:
            self.V[v].print_out()
    return