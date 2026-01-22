import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
class VertexCacheField(VertexCacheBase):

    def __init__(self, field=None, field_args=(), g_cons=None, g_cons_args=(), workers=1):
        """
        Class for a vertex cache for a simplicial complex with an associated
        field.

        Parameters
        ----------
        field : callable
            Scalar or vector field callable.
        field_args : tuple, optional
            Any additional fixed parameters needed to completely specify the
            field function
        g_cons : dict or sequence of dict, optional
            Constraints definition.
            Function(s) ``R**n`` in the form::
        g_cons_args : tuple, optional
            Any additional fixed parameters needed to completely specify the
            constraint functions
        workers : int  optional
            Uses `multiprocessing.Pool <multiprocessing>`) to compute the field
             functions in parallel.

        """
        super().__init__()
        self.index = -1
        self.Vertex = VertexScalarField
        self.field = field
        self.field_args = field_args
        self.wfield = FieldWrapper(field, field_args)
        self.g_cons = g_cons
        self.g_cons_args = g_cons_args
        self.wgcons = ConstraintWrapper(g_cons, g_cons_args)
        self.gpool = set()
        self.fpool = set()
        self.sfc_lock = False
        self.workers = workers
        self._mapwrapper = MapWrapper(workers)
        if workers == 1:
            self.process_gpool = self.proc_gpool
            if g_cons is None:
                self.process_fpool = self.proc_fpool_nog
            else:
                self.process_fpool = self.proc_fpool_g
        else:
            self.process_gpool = self.pproc_gpool
            if g_cons is None:
                self.process_fpool = self.pproc_fpool_nog
            else:
                self.process_fpool = self.pproc_fpool_g

    def __getitem__(self, x, nn=None):
        try:
            return self.cache[x]
        except KeyError:
            self.index += 1
            xval = self.Vertex(x, field=self.field, nn=nn, index=self.index, field_args=self.field_args, g_cons=self.g_cons, g_cons_args=self.g_cons_args)
            self.cache[x] = xval
            self.gpool.add(xval)
            self.fpool.add(xval)
            return self.cache[x]

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def process_pools(self):
        if self.g_cons is not None:
            self.process_gpool()
        self.process_fpool()
        self.proc_minimisers()

    def feasibility_check(self, v):
        v.feasible = True
        for g, args in zip(self.g_cons, self.g_cons_args):
            if np.any(g(v.x_a, *args) < 0.0):
                v.f = np.inf
                v.feasible = False
                break

    def compute_sfield(self, v):
        """Compute the scalar field values of a vertex object `v`.

        Parameters
        ----------
        v : VertexBase or VertexScalarField object
        """
        try:
            v.f = self.field(v.x_a, *self.field_args)
            self.nfev += 1
        except AttributeError:
            v.f = np.inf
        if np.isnan(v.f):
            v.f = np.inf

    def proc_gpool(self):
        """Process all constraints."""
        if self.g_cons is not None:
            for v in self.gpool:
                self.feasibility_check(v)
        self.gpool = set()

    def pproc_gpool(self):
        """Process all constraints in parallel."""
        gpool_l = []
        for v in self.gpool:
            gpool_l.append(v.x_a)
        G = self._mapwrapper(self.wgcons.gcons, gpool_l)
        for v, g in zip(self.gpool, G):
            v.feasible = g

    def proc_fpool_g(self):
        """Process all field functions with constraints supplied."""
        for v in self.fpool:
            if v.feasible:
                self.compute_sfield(v)
        self.fpool = set()

    def proc_fpool_nog(self):
        """Process all field functions with no constraints supplied."""
        for v in self.fpool:
            self.compute_sfield(v)
        self.fpool = set()

    def pproc_fpool_g(self):
        """
        Process all field functions with constraints supplied in parallel.
        """
        self.wfield.func
        fpool_l = []
        for v in self.fpool:
            if v.feasible:
                fpool_l.append(v.x_a)
            else:
                v.f = np.inf
        F = self._mapwrapper(self.wfield.func, fpool_l)
        for va, f in zip(fpool_l, F):
            vt = tuple(va)
            self[vt].f = f
            self.nfev += 1
        self.fpool = set()

    def pproc_fpool_nog(self):
        """
        Process all field functions with no constraints supplied in parallel.
        """
        self.wfield.func
        fpool_l = []
        for v in self.fpool:
            fpool_l.append(v.x_a)
        F = self._mapwrapper(self.wfield.func, fpool_l)
        for va, f in zip(fpool_l, F):
            vt = tuple(va)
            self[vt].f = f
            self.nfev += 1
        self.fpool = set()

    def proc_minimisers(self):
        """Check for minimisers."""
        for v in self:
            v.minimiser()
            v.maximiser()