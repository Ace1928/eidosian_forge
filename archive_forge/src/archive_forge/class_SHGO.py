from collections import namedtuple
import time
import logging
import warnings
import sys
import numpy as np
from scipy import spatial
from scipy.optimize import OptimizeResult, minimize, Bounds
from scipy.optimize._optimize import MemoizeJac
from scipy.optimize._constraints import new_bounds_to_old
from scipy.optimize._minimize import standardize_constraints
from scipy._lib._util import _FunctionWrapper
from scipy.optimize._shgo_lib._complex import Complex
class SHGO:

    def __init__(self, func, bounds, args=(), constraints=None, n=None, iters=None, callback=None, minimizer_kwargs=None, options=None, sampling_method='simplicial', workers=1):
        from scipy.stats import qmc
        methods = ['halton', 'sobol', 'simplicial']
        if isinstance(sampling_method, str) and sampling_method not in methods:
            raise ValueError('Unknown sampling_method specified. Valid methods: {}'.format(', '.join(methods)))
        try:
            if minimizer_kwargs['jac'] is True and (not callable(minimizer_kwargs['jac'])):
                self.func = MemoizeJac(func)
                jac = self.func.derivative
                minimizer_kwargs['jac'] = jac
                func = self.func
            else:
                self.func = func
        except (TypeError, KeyError):
            self.func = func
        self.func = _FunctionWrapper(func, args)
        self.bounds = bounds
        self.args = args
        self.callback = callback
        abound = np.array(bounds, float)
        self.dim = np.shape(abound)[0]
        infind = ~np.isfinite(abound)
        abound[infind[:, 0], 0] = -1e+50
        abound[infind[:, 1], 1] = 1e+50
        bnderr = abound[:, 0] > abound[:, 1]
        if bnderr.any():
            raise ValueError('Error: lb > ub in bounds {}.'.format(', '.join((str(b) for b in bnderr))))
        self.bounds = abound
        self.constraints = constraints
        if constraints is not None:
            self.min_cons = constraints
            self.g_cons = []
            self.g_args = []
            self.constraints = standardize_constraints(constraints, np.empty(self.dim, float), 'old')
            for cons in self.constraints:
                if cons['type'] in 'ineq':
                    self.g_cons.append(cons['fun'])
                    try:
                        self.g_args.append(cons['args'])
                    except KeyError:
                        self.g_args.append(())
            self.g_cons = tuple(self.g_cons)
            self.g_args = tuple(self.g_args)
        else:
            self.g_cons = None
            self.g_args = None
        self.minimizer_kwargs = {'method': 'SLSQP', 'bounds': self.bounds, 'options': {}, 'callback': self.callback}
        if minimizer_kwargs is not None:
            self.minimizer_kwargs.update(minimizer_kwargs)
        else:
            self.minimizer_kwargs['options'] = {'ftol': 1e-12}
        if self.minimizer_kwargs['method'].lower() in ('slsqp', 'cobyla', 'trust-constr') and (minimizer_kwargs is not None and 'constraints' not in minimizer_kwargs and (constraints is not None)) or self.g_cons is not None:
            self.minimizer_kwargs['constraints'] = self.min_cons
        if options is not None:
            self.init_options(options)
        else:
            self.f_min_true = None
            self.minimize_every_iter = True
            self.maxiter = None
            self.maxfev = None
            self.maxev = None
            self.maxtime = None
            self.f_min_true = None
            self.minhgrd = None
            self.symmetry = None
            self.infty_cons_sampl = True
            self.local_iter = False
            self.disp = False
        self.min_solver_args = ['fun', 'x0', 'args', 'callback', 'options', 'method']
        solver_args = {'_custom': ['jac', 'hess', 'hessp', 'bounds', 'constraints'], 'nelder-mead': [], 'powell': [], 'cg': ['jac'], 'bfgs': ['jac'], 'newton-cg': ['jac', 'hess', 'hessp'], 'l-bfgs-b': ['jac', 'bounds'], 'tnc': ['jac', 'bounds'], 'cobyla': ['constraints', 'catol'], 'slsqp': ['jac', 'bounds', 'constraints'], 'dogleg': ['jac', 'hess'], 'trust-ncg': ['jac', 'hess', 'hessp'], 'trust-krylov': ['jac', 'hess', 'hessp'], 'trust-exact': ['jac', 'hess'], 'trust-constr': ['jac', 'hess', 'hessp', 'constraints']}
        method = self.minimizer_kwargs['method']
        self.min_solver_args += solver_args[method.lower()]

        def _restrict_to_keys(dictionary, goodkeys):
            """Remove keys from dictionary if not in goodkeys - inplace"""
            existingkeys = set(dictionary)
            for key in existingkeys - set(goodkeys):
                dictionary.pop(key, None)
        _restrict_to_keys(self.minimizer_kwargs, self.min_solver_args)
        _restrict_to_keys(self.minimizer_kwargs['options'], self.min_solver_args + ['ftol'])
        self.stop_global = False
        self.break_routine = False
        self.iters = iters
        self.iters_done = 0
        self.n = n
        self.nc = 0
        self.n_prc = 0
        self.n_sampled = 0
        self.fn = 0
        self.hgr = 0
        self.qhull_incremental = True
        if self.n is None and self.iters is None and (sampling_method == 'simplicial'):
            self.n = 2 ** self.dim + 1
            self.nc = 0
        if self.iters is None:
            self.iters = 1
        if self.n is None and (not sampling_method == 'simplicial'):
            self.n = self.n = 100
            self.nc = 0
        if self.n == 100 and sampling_method == 'simplicial':
            self.n = 2 ** self.dim + 1
        if not (self.maxiter is None and self.maxfev is None and (self.maxev is None) and (self.minhgrd is None) and (self.f_min_true is None)):
            self.iters = None
        self.HC = Complex(dim=self.dim, domain=self.bounds, sfield=self.func, sfield_args=(), symmetry=self.symmetry, constraints=self.constraints, workers=workers)
        if sampling_method == 'simplicial':
            self.iterate_complex = self.iterate_hypercube
            self.sampling_method = sampling_method
        elif sampling_method in ['halton', 'sobol'] or not isinstance(sampling_method, str):
            self.iterate_complex = self.iterate_delaunay
            if sampling_method in ['halton', 'sobol']:
                if sampling_method == 'sobol':
                    self.n = int(2 ** np.ceil(np.log2(self.n)))
                    self.nc = 0
                    self.sampling_method = 'sobol'
                    self.qmc_engine = qmc.Sobol(d=self.dim, scramble=False, seed=0)
                else:
                    self.sampling_method = 'halton'
                    self.qmc_engine = qmc.Halton(d=self.dim, scramble=True, seed=0)

                def sampling_method(n, d):
                    return self.qmc_engine.random(n)
            else:
                self.sampling_method = 'custom'
            self.sampling = self.sampling_custom
            self.sampling_function = sampling_method
        self.stop_l_iter = False
        self.stop_complex_iter = False
        self.minimizer_pool = []
        self.LMC = LMapCache()
        self.res = OptimizeResult()
        self.res.nfev = 0
        self.res.nlfev = 0
        self.res.nljev = 0
        self.res.nlhev = 0

    def init_options(self, options):
        """
        Initiates the options.

        Can also be useful to change parameters after class initiation.

        Parameters
        ----------
        options : dict

        Returns
        -------
        None

        """
        self.minimizer_kwargs['options'].update(options)
        for opt in ['jac', 'hess', 'hessp']:
            if opt in self.minimizer_kwargs['options']:
                self.minimizer_kwargs[opt] = self.minimizer_kwargs['options'].pop(opt)
        self.minimize_every_iter = options.get('minimize_every_iter', True)
        self.maxiter = options.get('maxiter', None)
        self.maxfev = options.get('maxfev', None)
        self.maxev = options.get('maxev', None)
        self.init = time.time()
        self.maxtime = options.get('maxtime', None)
        if 'f_min' in options:
            self.f_min_true = options['f_min']
            self.f_tol = options.get('f_tol', 0.0001)
        else:
            self.f_min_true = None
        self.minhgrd = options.get('minhgrd', None)
        self.symmetry = options.get('symmetry', False)
        if self.symmetry:
            self.symmetry = [0] * len(self.bounds)
        else:
            self.symmetry = None
        self.local_iter = options.get('local_iter', False)
        self.infty_cons_sampl = options.get('infty_constraints', True)
        self.disp = options.get('disp', False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.HC.V._mapwrapper.__exit__(*args)

    def iterate_all(self):
        """
        Construct for `iters` iterations.

        If uniform sampling is used, every iteration adds 'n' sampling points.

        Iterations if a stopping criteria (e.g., sampling points or
        processing time) has been met.

        """
        if self.disp:
            logging.info('Splitting first generation')
        while not self.stop_global:
            if self.break_routine:
                break
            self.iterate()
            self.stopping_criteria()
        if not self.minimize_every_iter:
            if not self.break_routine:
                self.find_minima()
        self.res.nit = self.iters_done
        self.fn = self.HC.V.nfev

    def find_minima(self):
        """
        Construct the minimizer pool, map the minimizers to local minima
        and sort the results into a global return object.
        """
        if self.disp:
            logging.info('Searching for minimizer pool...')
        self.minimizers()
        if len(self.X_min) != 0:
            self.minimise_pool(self.local_iter)
            self.sort_result()
            self.f_lowest = self.res.fun
            self.x_lowest = self.res.x
        else:
            self.find_lowest_vertex()
        if self.disp:
            logging.info(f'Minimiser pool = SHGO.X_min = {self.X_min}')

    def find_lowest_vertex(self):
        self.f_lowest = np.inf
        for x in self.HC.V.cache:
            if self.HC.V[x].f < self.f_lowest:
                if self.disp:
                    logging.info(f'self.HC.V[x].f = {self.HC.V[x].f}')
                self.f_lowest = self.HC.V[x].f
                self.x_lowest = self.HC.V[x].x_a
        for lmc in self.LMC.cache:
            if self.LMC[lmc].f_min < self.f_lowest:
                self.f_lowest = self.LMC[lmc].f_min
                self.x_lowest = self.LMC[lmc].x_l
        if self.f_lowest == np.inf:
            self.f_lowest = None
            self.x_lowest = None

    def finite_iterations(self):
        mi = min((x for x in [self.iters, self.maxiter] if x is not None))
        if self.disp:
            logging.info(f'Iterations done = {self.iters_done} / {mi}')
        if self.iters is not None:
            if self.iters_done >= self.iters:
                self.stop_global = True
        if self.maxiter is not None:
            if self.iters_done >= self.maxiter:
                self.stop_global = True
        return self.stop_global

    def finite_fev(self):
        if self.disp:
            logging.info(f'Function evaluations done = {self.fn} / {self.maxfev}')
        if self.fn >= self.maxfev:
            self.stop_global = True
        return self.stop_global

    def finite_ev(self):
        if self.disp:
            logging.info(f'Sampling evaluations done = {self.n_sampled} / {self.maxev}')
        if self.n_sampled >= self.maxev:
            self.stop_global = True

    def finite_time(self):
        if self.disp:
            logging.info(f'Time elapsed = {time.time() - self.init} / {self.maxtime}')
        if time.time() - self.init >= self.maxtime:
            self.stop_global = True

    def finite_precision(self):
        """
        Stop the algorithm if the final function value is known

        Specify in options (with ``self.f_min_true = options['f_min']``)
        and the tolerance with ``f_tol = options['f_tol']``
        """
        self.find_lowest_vertex()
        if self.disp:
            logging.info(f'Lowest function evaluation = {self.f_lowest}')
            logging.info(f'Specified minimum = {self.f_min_true}')
        if self.f_lowest is None:
            return self.stop_global
        if self.f_min_true == 0.0:
            if self.f_lowest <= self.f_tol:
                self.stop_global = True
        else:
            pe = (self.f_lowest - self.f_min_true) / abs(self.f_min_true)
            if self.f_lowest <= self.f_min_true:
                self.stop_global = True
                if abs(pe) >= 2 * self.f_tol:
                    warnings.warn(f'A much lower value than expected f* = {self.f_min_true} was found f_lowest = {self.f_lowest}', stacklevel=3)
            if pe <= self.f_tol:
                self.stop_global = True
        return self.stop_global

    def finite_homology_growth(self):
        """
        Stop the algorithm if homology group rank did not grow in iteration.
        """
        if self.LMC.size == 0:
            return
        self.hgrd = self.LMC.size - self.hgr
        self.hgr = self.LMC.size
        if self.hgrd <= self.minhgrd:
            self.stop_global = True
        if self.disp:
            logging.info(f'Current homology growth = {self.hgrd}  (minimum growth = {self.minhgrd})')
        return self.stop_global

    def stopping_criteria(self):
        """
        Various stopping criteria ran every iteration

        Returns
        -------
        stop : bool
        """
        if self.maxiter is not None:
            self.finite_iterations()
        if self.iters is not None:
            self.finite_iterations()
        if self.maxfev is not None:
            self.finite_fev()
        if self.maxev is not None:
            self.finite_ev()
        if self.maxtime is not None:
            self.finite_time()
        if self.f_min_true is not None:
            self.finite_precision()
        if self.minhgrd is not None:
            self.finite_homology_growth()
        return self.stop_global

    def iterate(self):
        self.iterate_complex()
        if self.minimize_every_iter:
            if not self.break_routine:
                self.find_minima()
        self.iters_done += 1

    def iterate_hypercube(self):
        """
        Iterate a subdivision of the complex

        Note: called with ``self.iterate_complex()`` after class initiation
        """
        if self.disp:
            logging.info('Constructing and refining simplicial complex graph structure')
        if self.n is None:
            self.HC.refine_all()
            self.n_sampled = self.HC.V.size()
        else:
            self.HC.refine(self.n)
            self.n_sampled += self.n
        if self.disp:
            logging.info('Triangulation completed, evaluating all constraints and objective function values.')
        if len(self.LMC.xl_maps) > 0:
            for xl in self.LMC.cache:
                v = self.HC.V[xl]
                v_near = v.star()
                for v in v.nn:
                    v_near = v_near.union(v.nn)
        self.HC.V.process_pools()
        if self.disp:
            logging.info('Evaluations completed.')
        self.fn = self.HC.V.nfev
        return

    def iterate_delaunay(self):
        """
        Build a complex of Delaunay triangulated points

        Note: called with ``self.iterate_complex()`` after class initiation
        """
        self.nc += self.n
        self.sampled_surface(infty_cons_sampl=self.infty_cons_sampl)
        if self.disp:
            logging.info(f'self.n = {self.n}')
            logging.info(f'self.nc = {self.nc}')
            logging.info('Constructing and refining simplicial complex graph structure from sampling points.')
        if self.dim < 2:
            self.Ind_sorted = np.argsort(self.C, axis=0)
            self.Ind_sorted = self.Ind_sorted.flatten()
            tris = []
            for ind, ind_s in enumerate(self.Ind_sorted):
                if ind > 0:
                    tris.append(self.Ind_sorted[ind - 1:ind + 1])
            tris = np.array(tris)
            self.Tri = namedtuple('Tri', ['points', 'simplices'])(self.C, tris)
            self.points = {}
        else:
            if self.C.shape[0] > self.dim + 1:
                self.delaunay_triangulation(n_prc=self.n_prc)
            self.n_prc = self.C.shape[0]
        if self.disp:
            logging.info('Triangulation completed, evaluating all constraints and objective function values.')
        if hasattr(self, 'Tri'):
            self.HC.vf_to_vv(self.Tri.points, self.Tri.simplices)
        if self.disp:
            logging.info('Triangulation completed, evaluating all constraints and objective function values.')
        self.HC.V.process_pools()
        if self.disp:
            logging.info('Evaluations completed.')
        self.fn = self.HC.V.nfev
        self.n_sampled = self.nc
        return

    def minimizers(self):
        """
        Returns the indexes of all minimizers
        """
        self.minimizer_pool = []
        for x in self.HC.V.cache:
            in_LMC = False
            if len(self.LMC.xl_maps) > 0:
                for xlmi in self.LMC.xl_maps:
                    if np.all(np.array(x) == np.array(xlmi)):
                        in_LMC = True
            if in_LMC:
                continue
            if self.HC.V[x].minimiser():
                if self.disp:
                    logging.info('=' * 60)
                    logging.info(f'v.x = {self.HC.V[x].x_a} is minimizer')
                    logging.info(f'v.f = {self.HC.V[x].f} is minimizer')
                    logging.info('=' * 30)
                if self.HC.V[x] not in self.minimizer_pool:
                    self.minimizer_pool.append(self.HC.V[x])
                if self.disp:
                    logging.info('Neighbors:')
                    logging.info('=' * 30)
                    for vn in self.HC.V[x].nn:
                        logging.info(f'x = {vn.x} || f = {vn.f}')
                    logging.info('=' * 60)
        self.minimizer_pool_F = []
        self.X_min = []
        self.X_min_cache = {}
        for v in self.minimizer_pool:
            self.X_min.append(v.x_a)
            self.minimizer_pool_F.append(v.f)
            self.X_min_cache[tuple(v.x_a)] = v.x
        self.minimizer_pool_F = np.array(self.minimizer_pool_F)
        self.X_min = np.array(self.X_min)
        self.sort_min_pool()
        return self.X_min

    def minimise_pool(self, force_iter=False):
        """
        This processing method can optionally minimise only the best candidate
        solutions in the minimiser pool

        Parameters
        ----------
        force_iter : int
                     Number of starting minimizers to process (can be specified
                     globally or locally)

        """
        lres_f_min = self.minimize(self.X_min[0], ind=self.minimizer_pool[0])
        self.trim_min_pool(0)
        while not self.stop_l_iter:
            self.stopping_criteria()
            if force_iter:
                force_iter -= 1
                if force_iter == 0:
                    self.stop_l_iter = True
                    break
            if np.shape(self.X_min)[0] == 0:
                self.stop_l_iter = True
                break
            self.g_topograph(lres_f_min.x, self.X_min)
            ind_xmin_l = self.Z[:, -1]
            lres_f_min = self.minimize(self.Ss[-1, :], self.minimizer_pool[-1])
            self.trim_min_pool(ind_xmin_l)
        self.stop_l_iter = False
        return

    def sort_min_pool(self):
        self.ind_f_min = np.argsort(self.minimizer_pool_F)
        self.minimizer_pool = np.array(self.minimizer_pool)[self.ind_f_min]
        self.minimizer_pool_F = np.array(self.minimizer_pool_F)[self.ind_f_min]
        return

    def trim_min_pool(self, trim_ind):
        self.X_min = np.delete(self.X_min, trim_ind, axis=0)
        self.minimizer_pool_F = np.delete(self.minimizer_pool_F, trim_ind)
        self.minimizer_pool = np.delete(self.minimizer_pool, trim_ind)
        return

    def g_topograph(self, x_min, X_min):
        """
        Returns the topographical vector stemming from the specified value
        ``x_min`` for the current feasible set ``X_min`` with True boolean
        values indicating positive entries and False values indicating
        negative entries.

        """
        x_min = np.array([x_min])
        self.Y = spatial.distance.cdist(x_min, X_min, 'euclidean')
        self.Z = np.argsort(self.Y, axis=-1)
        self.Ss = X_min[self.Z][0]
        self.minimizer_pool = self.minimizer_pool[self.Z]
        self.minimizer_pool = self.minimizer_pool[0]
        return self.Ss

    def construct_lcb_simplicial(self, v_min):
        """
        Construct locally (approximately) convex bounds

        Parameters
        ----------
        v_min : Vertex object
                The minimizer vertex

        Returns
        -------
        cbounds : list of lists
            List of size dimension with length-2 list of bounds for each
            dimension.

        """
        cbounds = [[x_b_i[0], x_b_i[1]] for x_b_i in self.bounds]
        for vn in v_min.nn:
            for i, x_i in enumerate(vn.x_a):
                if x_i < v_min.x_a[i] and x_i > cbounds[i][0]:
                    cbounds[i][0] = x_i
                if x_i > v_min.x_a[i] and x_i < cbounds[i][1]:
                    cbounds[i][1] = x_i
        if self.disp:
            logging.info(f'cbounds found for v_min.x_a = {v_min.x_a}')
            logging.info(f'cbounds = {cbounds}')
        return cbounds

    def construct_lcb_delaunay(self, v_min, ind=None):
        """
        Construct locally (approximately) convex bounds

        Parameters
        ----------
        v_min : Vertex object
                The minimizer vertex

        Returns
        -------
        cbounds : list of lists
            List of size dimension with length-2 list of bounds for each
            dimension.
        """
        cbounds = [[x_b_i[0], x_b_i[1]] for x_b_i in self.bounds]
        return cbounds

    def minimize(self, x_min, ind=None):
        """
        This function is used to calculate the local minima using the specified
        sampling point as a starting value.

        Parameters
        ----------
        x_min : vector of floats
            Current starting point to minimize.

        Returns
        -------
        lres : OptimizeResult
            The local optimization result represented as a `OptimizeResult`
            object.
        """
        if self.disp:
            logging.info(f'Vertex minimiser maps = {self.LMC.v_maps}')
        if self.LMC[x_min].lres is not None:
            logging.info(f'Found self.LMC[x_min].lres = {self.LMC[x_min].lres}')
            return self.LMC[x_min].lres
        if self.callback is not None:
            logging.info(f'Callback for minimizer starting at {x_min}:')
        if self.disp:
            logging.info(f'Starting minimization at {x_min}...')
        if self.sampling_method == 'simplicial':
            x_min_t = tuple(x_min)
            x_min_t_norm = self.X_min_cache[tuple(x_min_t)]
            x_min_t_norm = tuple(x_min_t_norm)
            g_bounds = self.construct_lcb_simplicial(self.HC.V[x_min_t_norm])
            if 'bounds' in self.min_solver_args:
                self.minimizer_kwargs['bounds'] = g_bounds
                logging.info(self.minimizer_kwargs['bounds'])
        else:
            g_bounds = self.construct_lcb_delaunay(x_min, ind=ind)
            if 'bounds' in self.min_solver_args:
                self.minimizer_kwargs['bounds'] = g_bounds
                logging.info(self.minimizer_kwargs['bounds'])
        if self.disp and 'bounds' in self.minimizer_kwargs:
            logging.info('bounds in kwarg:')
            logging.info(self.minimizer_kwargs['bounds'])
        lres = minimize(self.func, x_min, **self.minimizer_kwargs)
        if self.disp:
            logging.info(f'lres = {lres}')
        self.res.nlfev += lres.nfev
        if 'njev' in lres:
            self.res.nljev += lres.njev
        if 'nhev' in lres:
            self.res.nlhev += lres.nhev
        try:
            lres.fun = lres.fun[0]
        except (IndexError, TypeError):
            lres.fun
        self.LMC[x_min]
        self.LMC.add_res(x_min, lres, bounds=g_bounds)
        return lres

    def sort_result(self):
        """
        Sort results and build the global return object
        """
        results = self.LMC.sort_cache_result()
        self.res.xl = results['xl']
        self.res.funl = results['funl']
        self.res.x = results['x']
        self.res.fun = results['fun']
        self.res.nfev = self.fn + self.res.nlfev
        return self.res

    def fail_routine(self, mes='Failed to converge'):
        self.break_routine = True
        self.res.success = False
        self.X_min = [None]
        self.res.message = mes

    def sampled_surface(self, infty_cons_sampl=False):
        """
        Sample the function surface.

        There are 2 modes, if ``infty_cons_sampl`` is True then the sampled
        points that are generated outside the feasible domain will be
        assigned an ``inf`` value in accordance with SHGO rules.
        This guarantees convergence and usually requires less objective
        function evaluations at the computational costs of more Delaunay
        triangulation points.

        If ``infty_cons_sampl`` is False, then the infeasible points are
        discarded and only a subspace of the sampled points are used. This
        comes at the cost of the loss of guaranteed convergence and usually
        requires more objective function evaluations.
        """
        if self.disp:
            logging.info('Generating sampling points')
        self.sampling(self.nc, self.dim)
        if len(self.LMC.xl_maps) > 0:
            self.C = np.vstack((self.C, np.array(self.LMC.xl_maps)))
        if not infty_cons_sampl:
            if self.g_cons is not None:
                self.sampling_subspace()
        self.sorted_samples()
        self.n_sampled = self.nc

    def sampling_custom(self, n, dim):
        """
        Generates uniform sampling points in a hypercube and scales the points
        to the bound limits.
        """
        if self.n_sampled == 0:
            self.C = self.sampling_function(n, dim)
        else:
            self.C = self.sampling_function(n, dim)
        for i in range(len(self.bounds)):
            self.C[:, i] = self.C[:, i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]
        return self.C

    def sampling_subspace(self):
        """Find subspace of feasible points from g_func definition"""
        for ind, g in enumerate(self.g_cons):
            feasible = np.array([np.all(g(x_C, *self.g_args[ind]) >= 0.0) for x_C in self.C], dtype=bool)
            self.C = self.C[feasible]
            if self.C.size == 0:
                self.res.message = 'No sampling point found within the ' + 'feasible set. Increasing sampling ' + 'size.'
                if self.disp:
                    logging.info(self.res.message)

    def sorted_samples(self):
        """Find indexes of the sorted sampling points"""
        self.Ind_sorted = np.argsort(self.C, axis=0)
        self.Xs = self.C[self.Ind_sorted]
        return (self.Ind_sorted, self.Xs)

    def delaunay_triangulation(self, n_prc=0):
        if hasattr(self, 'Tri') and self.qhull_incremental:
            self.Tri.add_points(self.C[n_prc:, :])
        else:
            try:
                self.Tri = spatial.Delaunay(self.C, incremental=self.qhull_incremental)
            except spatial.QhullError:
                if str(sys.exc_info()[1])[:6] == 'QH6239':
                    logging.warning('QH6239 Qhull precision error detected, this usually occurs when no bounds are specified, Qhull can only run with handling cocircular/cospherical points and in this case incremental mode is switched off. The performance of shgo will be reduced in this mode.')
                    self.qhull_incremental = False
                    self.Tri = spatial.Delaunay(self.C, incremental=self.qhull_incremental)
                else:
                    raise
        return self.Tri