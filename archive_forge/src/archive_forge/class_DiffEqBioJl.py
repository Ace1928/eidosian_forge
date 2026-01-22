from collections import OrderedDict
from chempy import Reaction
from chempy.kinetics.rates import MassAction, RadiolyticBase
from chempy.units import to_unitless, default_units as u
class DiffEqBioJl:
    _template_body = '{name} = @{crn_macro} begin\n    {reactions}\nend {parameters}\n{post}\n'
    defaults = dict(unit_conc=u.molar, unit_time=u.second)

    def __init__(self, *, rxs, pars, substance_key_map, parmap, **kwargs):
        self.rxs = rxs
        self.pars = pars
        self.substance_key_map = substance_key_map
        self.parmap = parmap
        self.unit_conc = kwargs.get('unit_conc', self.defaults['unit_conc'])
        self.unit_time = kwargs.get('unit_time', self.defaults['unit_time'])

    @classmethod
    def from_rsystem(cls, rsys, par_vals, *, variables=None, substance_key_map=lambda i, sk: 'y%d' % i, **kwargs):
        if not isinstance(substance_key_map, dict):
            substance_key_map = {sk: substance_key_map(si, sk) for si, sk in enumerate(rsys.substances)}
        parmap = dict([(r.param.unique_keys[0], 'p%d' % i) for i, r in enumerate(rsys.rxns)])
        rxs, pars = ([], OrderedDict())
        for r in rsys.rxns:
            rs, pk, pv = _r(r, par_vals, substance_key_map, parmap, variables=variables, unit_conc=kwargs.get('unit_conc', cls.defaults['unit_conc']), unit_time=kwargs.get('unit_time', cls.defaults['unit_time']))
            rxs.append(rs)
            if pk in pars:
                raise ValueError('Are you sure (sometimes intentional)?')
            pars[parmap[pk]] = pv
        return cls(rxs=rxs, pars=pars, substance_key_map=substance_key_map, parmap=parmap, **kwargs)

    def render_body(self, sparse_jac=False):
        name = 'rn'
        return self._template_body.format(crn_macro='min_reaction_network' if sparse_jac else 'reaction_network', name=name, reactions='\n    '.join(self.rxs), parameters=' '.join(self.pars), post='addodes!({}, sparse_jac=True)'.format(name) if sparse_jac else '')

    def render_setup(self, *, ics, atol, tex=True, tspan=None):
        export = ''
        export += 'p = %s\n' % jl_dict(self.pars)
        export += 'ics = %s\n' % jl_dict(OrderedDict({self.substance_key_map[k]: v for k, v in to_unitless(ics, u.molar).items()}))
        if atol:
            export += 'abstol_d = %s\n' % jl_dict({self.substance_key_map[k]: v for k, v in to_unitless(atol, u.molar).items()})
            export += 'abstol = Array([get(abstol_d, k, 1e-10) for k=keys(speciesmap(rn))])'
        if tex:
            export += 'subst_tex = Dict([%s])\n' % ', '.join(('(:%s, ("%s", "%s"))' % (v, k, k.latex_name) for k, v in self.substance_key_map.items()))
        if tspan:
            export += 'tspan = (0., %12.5g)\nu0 = Array([get(ics, k, 1e-28) for k=keys(speciesmap(rn))])\nparr = Array([p[k] for k=keys(paramsmap(rn))])\noprob = ODEProblem(rn, u0, tspan, parr)\n'
        return export

    def render_solve(self):
        return 'sol = solve(oprob, reltol=1e-9, abstol=abstol, Rodas5(), callback=PositiveDomain(ones(length(u0)), abstol=abstol))'