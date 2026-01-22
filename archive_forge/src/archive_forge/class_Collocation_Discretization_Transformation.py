import logging
import math
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.collections import ComponentSet
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core import Var, ConstraintList, Expression, Objective
from pyomo.dae import ContinuousSet, DerivativeVar, Integral
from pyomo.dae.misc import generate_finite_elements
from pyomo.dae.misc import generate_colloc_points
from pyomo.dae.misc import expand_components
from pyomo.dae.misc import create_partial_expression
from pyomo.dae.misc import add_discretization_equations
from pyomo.dae.misc import add_continuity_equations
from pyomo.dae.misc import block_fully_discretized
from pyomo.dae.misc import get_index_information
from pyomo.dae.diffvar import DAE_Error
from pyomo.common.config import ConfigBlock, ConfigValue, PositiveInt, In
@TransformationFactory.register('dae.collocation', doc='Discretizes a DAE model using orthogonal collocation over finite elements transforming the model into an NLP.')
class Collocation_Discretization_Transformation(Transformation):
    CONFIG = ConfigBlock('dae.collocation')
    CONFIG.declare('nfe', ConfigValue(default=10, domain=PositiveInt, description='The desired number of finite element points to be included in the discretization'))
    CONFIG.declare('ncp', ConfigValue(default=3, domain=PositiveInt, description='The desired number of collocation points over each finite element'))
    CONFIG.declare('wrt', ConfigValue(default=None, description='The ContinuousSet to be discretized', doc='Indicates which ContinuousSet the transformation should be applied to. If this keyword argument is not specified then the same scheme will be applied to all ContinuousSets.'))
    CONFIG.declare('scheme', ConfigValue(default='LAGRANGE-RADAU', domain=In(['LAGRANGE-RADAU', 'LAGRANGE-LEGENDRE']), description='Indicates which collocation scheme to apply', doc="Options are 'LAGRANGE-RADAU' and 'LAGRANGE-LEGENDRE'. The default scheme is Lagrange polynomials with Radau roots"))

    def __init__(self):
        super(Collocation_Discretization_Transformation, self).__init__()
        self._ncp = {}
        self._nfe = {}
        self._adot = {}
        self._adotdot = {}
        self._afinal = {}
        self._tau = {}
        self._reduced_cp = {}
        self.all_schemes = {'LAGRANGE-RADAU': (_lagrange_radau_transform, _lagrange_radau_transform_order2), 'LAGRANGE-LEGENDRE': (_lagrange_legendre_transform, _lagrange_legendre_transform_order2)}

    def _get_radau_constants(self, currentds):
        """
        This function sets the radau collocation points and a values depending
        on how many collocation points have been specified and whether or not
        the user has numpy
        """
        if not numpy_available:
            if self._ncp[currentds] > 10:
                raise ValueError('Numpy was not found so the maximum number of collocation points is 10')
            from pyomo.dae.utilities import radau_tau_dict, radau_adot_dict, radau_adotdot_dict
            self._tau[currentds] = radau_tau_dict[self._ncp[currentds]]
            self._adot[currentds] = radau_adot_dict[self._ncp[currentds]]
            self._adotdot[currentds] = radau_adotdot_dict[self._ncp[currentds]]
            self._afinal[currentds] = None
        else:
            alpha = 1
            beta = 0
            k = self._ncp[currentds] - 1
            cp = calc_cp(alpha, beta, k)
            cp.insert(0, 0.0)
            cp.append(1.0)
            adot = calc_adot(cp, 1)
            adotdot = calc_adot(cp, 2)
            self._tau[currentds] = cp
            self._adot[currentds] = adot
            self._adotdot[currentds] = adotdot
            self._afinal[currentds] = None

    def _get_legendre_constants(self, currentds):
        """
        This function sets the legendre collocation points and a values
        depending on how many collocation points have been specified and
        whether or not the user has numpy
        """
        if not numpy_available:
            if self._ncp[currentds] > 10:
                raise ValueError('Numpy was not found so the maximum number of collocation points is 10')
            from pyomo.dae.utilities import legendre_tau_dict, legendre_adot_dict, legendre_adotdot_dict, legendre_afinal_dict
            self._tau[currentds] = legendre_tau_dict[self._ncp[currentds]]
            self._adot[currentds] = legendre_adot_dict[self._ncp[currentds]]
            self._adotdot[currentds] = legendre_adotdot_dict[self._ncp[currentds]]
            self._afinal[currentds] = legendre_afinal_dict[self._ncp[currentds]]
        else:
            alpha = 0
            beta = 0
            k = self._ncp[currentds]
            cp = calc_cp(alpha, beta, k)
            cp.insert(0, 0.0)
            adot = calc_adot(cp, 1)
            adotdot = calc_adot(cp, 2)
            afinal = calc_afinal(cp)
            self._tau[currentds] = cp
            self._adot[currentds] = adot
            self._adotdot[currentds] = adotdot
            self._afinal[currentds] = afinal

    def _apply_to(self, instance, **kwds):
        """
        Applies specified collocation transformation to a modeling instance

        Keyword Arguments:
        nfe           The desired number of finite element points to be
                      included in the discretization.
        ncp           The desired number of collocation points over each
                      finite element.
        wrt           Indicates which ContinuousSet the transformation
                      should be applied to. If this keyword argument is not
                      specified then the same scheme will be applied to all
                      ContinuousSets.
        scheme        Indicates which collocation scheme to apply.
                      Options are 'LAGRANGE-RADAU' and 'LAGRANGE-LEGENDRE'.
                      The default scheme is Lagrange polynomials with Radau
                      roots.
        """
        config = self.CONFIG(kwds)
        tmpnfe = config.nfe
        tmpncp = config.ncp
        tmpds = config.wrt
        if tmpds is not None:
            if tmpds.ctype is not ContinuousSet:
                raise TypeError("The component specified using the 'wrt' keyword must be a continuous set")
            elif 'scheme' in tmpds.get_discretization_info():
                raise ValueError("The discretization scheme '%s' has already been applied to the ContinuousSet '%s'" % (tmpds.get_discretization_info()['scheme'], tmpds.name))
        if None in self._nfe:
            raise ValueError('A general discretization scheme has already been applied to to every ContinuousSet in the model. If you would like to specify a specific discretization scheme for one of the ContinuousSets you must discretize each ContinuousSet separately.')
        if len(self._nfe) == 0 and tmpds is None:
            self._nfe[None] = tmpnfe
            self._ncp[None] = tmpncp
            currentds = None
        else:
            self._nfe[tmpds.name] = tmpnfe
            self._ncp[tmpds.name] = tmpncp
            currentds = tmpds.name
        self._scheme_name = config.scheme
        self._scheme = self.all_schemes.get(self._scheme_name, None)
        if self._scheme_name == 'LAGRANGE-RADAU':
            self._get_radau_constants(currentds)
        elif self._scheme_name == 'LAGRANGE-LEGENDRE':
            self._get_legendre_constants(currentds)
        self._transformBlock(instance, currentds)

    def _transformBlock(self, block, currentds):
        self._fe = {}
        for ds in block.component_objects(ContinuousSet, descend_into=True):
            if currentds is None or currentds == ds.name:
                if 'scheme' in ds.get_discretization_info():
                    raise DAE_Error("Attempting to discretize ContinuousSet '%s' after it has already been discretized. " % ds.name)
                generate_finite_elements(ds, self._nfe[currentds])
                if not ds.get_changed():
                    if len(ds) - 1 > self._nfe[currentds]:
                        logger.warning("More finite elements were found in ContinuousSet '%s' than the number of finite elements specified in apply. The larger number of finite elements will be used." % ds.name)
                self._nfe[ds.name] = len(ds) - 1
                self._fe[ds.name] = list(ds)
                generate_colloc_points(ds, self._tau[currentds])
                disc_info = ds.get_discretization_info()
                disc_info['nfe'] = self._nfe[ds.name]
                disc_info['ncp'] = self._ncp[currentds]
                disc_info['tau_points'] = self._tau[currentds]
                disc_info['adot'] = self._adot[currentds]
                disc_info['adotdot'] = self._adotdot[currentds]
                disc_info['afinal'] = self._afinal[currentds]
                disc_info['scheme'] = self._scheme_name
        expand_components(block)
        for d in block.component_objects(DerivativeVar, descend_into=True):
            dsets = d.get_continuousset_list()
            for i in ComponentSet(dsets):
                if currentds is None or i.name == currentds:
                    oldexpr = d.get_derivative_expression()
                    loc = d.get_state_var()._contset[i]
                    count = dsets.count(i)
                    if count >= 3:
                        raise DAE_Error("Error discretizing '%s' with respect to '%s'. Current implementation only allows for taking the first or second derivative with respect to a particular ContinuousSet" % (d.name, i.name))
                    scheme = self._scheme[count - 1]
                    newexpr = create_partial_expression(scheme, oldexpr, i, loc)
                    d.set_derivative_expression(newexpr)
                    if self._scheme_name == 'LAGRANGE-LEGENDRE':
                        add_continuity_equations(d.parent_block(), d, i, loc)
            if d.is_fully_discretized():
                add_discretization_equations(d.parent_block(), d)
                d.parent_block().reclassify_component_type(d, Var)
                reclassified_list = getattr(block, '_pyomo_dae_reclassified_derivativevars', None)
                if reclassified_list is None:
                    block._pyomo_dae_reclassified_derivativevars = list()
                    reclassified_list = block._pyomo_dae_reclassified_derivativevars
                reclassified_list.append(d)
        if block_fully_discretized(block):
            if block.contains_component(Integral):
                for i in block.component_objects(Integral, descend_into=True):
                    i.parent_block().reclassify_component_type(i, Expression)
                    i.clear()
                    i._constructed = False
                    i.construct()
                for k in block.component_objects(Objective, descend_into=True):
                    k.clear()
                    k._constructed = False
                    k.construct()

    def reduce_collocation_points(self, instance, var=None, ncp=None, contset=None):
        """
        This method will add additional constraints to a model to reduce the
        number of free collocation points (degrees of freedom) for a particular
        variable.

        Parameters
        ----------
        instance : Pyomo model
            The discretized Pyomo model to add constraints to

        var : ``pyomo.environ.Var``
            The Pyomo variable for which the degrees of freedom will be reduced

        ncp : int
            The new number of free collocation points for `var`. Must be
            less that the number of collocation points used in discretizing
            the model.

        contset : ``pyomo.dae.ContinuousSet``
            The :py:class:`ContinuousSet<pyomo.dae.ContinuousSet>` that was
            discretized and for which the `var` will have a reduced number
            of degrees of freedom

        """
        if contset is None:
            raise TypeError("A continuous set must be specified using the keyword 'contset'")
        if contset.ctype is not ContinuousSet:
            raise TypeError("The component specified using the 'contset' keyword must be a ContinuousSet")
        ds = contset
        if len(self._ncp) == 0:
            raise RuntimeError('This method should only be called after using the apply() method to discretize the model')
        elif None in self._ncp:
            tot_ncp = self._ncp[None]
        elif ds.name in self._ncp:
            tot_ncp = self._ncp[ds.name]
        else:
            raise ValueError("ContinuousSet '%s' has not been discretized, please call the apply_to() method with this ContinuousSet to discretize it before calling this method" % ds.name)
        if var is None:
            raise TypeError('A variable must be specified')
        if var.ctype is not Var:
            raise TypeError("The component specified using the 'var' keyword must be a variable")
        if ncp is None:
            raise TypeError('The number of collocation points must be specified')
        if ncp <= 0:
            raise ValueError('The number of collocation points must be at least 1')
        if ncp > tot_ncp:
            raise ValueError('The number of collocation points used to interpolate an individual variable must be less than the number used to discretize the original model')
        if ncp == tot_ncp:
            return instance
        if var.dim() == 0:
            raise IndexError("ContinuousSet '%s' is not an indexing set of the variable '%s'" % (ds.name, var.name))
        varidx = var.index_set()
        if not varidx.subsets():
            if ds is not varidx:
                raise IndexError("ContinuousSet '%s' is not an indexing set of the variable '%s'" % (ds.name, var.name))
        elif ds not in varidx.subsets():
            raise IndexError("ContinuousSet '%s' is not an indexing set of the variable '%s'" % (ds.name, var.name))
        if var.name in self._reduced_cp:
            temp = self._reduced_cp[var.name]
            if ds.name in temp:
                raise RuntimeError("Variable '%s' has already been constrained to a reduced number of collocation points over ContinuousSet '%s'.")
            else:
                temp[ds.name] = ncp
        else:
            self._reduced_cp[var.name] = {ds.name: ncp}
        list_name = var.local_name + '_interpolation_constraints'
        instance.add_component(list_name, ConstraintList())
        conlist = instance.find_component(list_name)
        t = list(ds)
        fe = ds._fe
        info = get_index_information(var, ds)
        tmpidx = info['non_ds']
        idx = info['index function']
        for n in tmpidx:
            for i in range(0, len(fe) - 1):
                for k in range(1, tot_ncp - ncp + 1):
                    if ncp == 1:
                        conlist.add(var[idx(n, i, k)] == var[idx(n, i, tot_ncp)])
                    else:
                        tmp = ds.ord(fe[i]) - 1
                        tmp2 = ds.ord(fe[i + 1]) - 1
                        ti = t[tmp + k]
                        tfit = t[tmp2 - ncp + 1:tmp2 + 1]
                        coeff = self._interpolation_coeffs(ti, tfit)
                        conlist.add(var[idx(n, i, k)] == sum((var[idx(n, i, j)] * next(coeff) for j in range(tot_ncp - ncp + 1, tot_ncp + 1))))
        return instance

    def _interpolation_coeffs(self, ti, tfit):
        for i in tfit:
            l = 1
            for j in tfit:
                if i != j:
                    l = l * (ti - j) / (i - j)
            yield l