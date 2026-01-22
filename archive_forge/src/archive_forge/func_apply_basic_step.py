import copy
import itertools
from pyomo.core import Block, ConstraintList, Set, Constraint
from pyomo.core.base import Reference
from pyomo.common.modeling import unique_component_name
from pyomo.gdp.disjunct import Disjunct, Disjunction
import logging
def apply_basic_step(disjunctions_or_constraints):
    disjunctions = list((obj for obj in disjunctions_or_constraints if obj.ctype is Disjunction))
    constraints = list((obj for obj in disjunctions_or_constraints if obj.ctype is Constraint))
    if len(disjunctions) + len(constraints) != len(disjunctions_or_constraints):
        raise ValueError('apply_basic_step only accepts a list containing Disjunctions or Constraints')
    if not disjunctions:
        raise ValueError('apply_basic_step: argument list must contain at least one Disjunction')
    for d in disjunctions:
        if not d.xor:
            raise ValueError("Basic steps can only be applied to XOR'd disjunctions\n\t(raised by disjunction %s)" % (d.name,))
        if not d.active:
            logger.warning('Warning: applying basic step to a previously deactivated disjunction (%s)' % (d.name,))
    ans = Block(concrete=True)
    ans.DISJUNCTIONS = Set(initialize=range(len(disjunctions)))
    ans.INDEX = Set(dimen=len(disjunctions), initialize=_squish_singletons(itertools.product(*tuple((range(len(d.disjuncts)) for d in disjunctions)))))
    ans.disjuncts = Disjunct(ans.INDEX)
    for idx in ans.INDEX:
        ans.disjuncts[idx].src = Block(ans.DISJUNCTIONS)
        for i in ans.DISJUNCTIONS:
            src_disj = disjunctions[i].disjuncts[idx[i] if isinstance(idx, tuple) else idx]
            tmp = _clone_all_but_indicator_vars(src_disj)
            for k, v in list(tmp.component_map().items()):
                if v.parent_block() is not tmp:
                    continue
                tmp.del_component(k)
                ans.disjuncts[idx].src[i].add_component(k, v)
        ans.disjuncts[idx].improper_constraints = ConstraintList()
        for constr in constraints:
            if constr.is_indexed():
                for indx in constr:
                    ans.disjuncts[idx].improper_constraints.add((constr[indx].lower, constr[indx].body, constr[indx].upper))
                    constr[indx].deactivate()
            else:
                ans.disjuncts[idx].improper_constraints.add((constr.lower, constr.body, constr.upper))
                constr.deactivate()
    ans.indicator_links = ConstraintList()
    for i in ans.DISJUNCTIONS:
        for j in range(len(disjunctions[i].disjuncts)):
            orig_var = disjunctions[i].disjuncts[j].indicator_var
            orig_binary_var = orig_var.get_associated_binary()
            ans.indicator_links.add(orig_binary_var == sum((ans.disjuncts[idx].binary_indicator_var for idx in ans.INDEX if (idx[i] if isinstance(idx, tuple) else idx) == j)))
            for v in (orig_var, orig_binary_var):
                name_base = v.getname(fully_qualified=True)
                ans.add_component(unique_component_name(ans, name_base), Reference(v))
    ans.disjunction = Disjunction(expr=[ans.disjuncts[i] for i in ans.INDEX])
    for i in ans.DISJUNCTIONS:
        disjunctions[i].deactivate()
        for d in disjunctions[i].disjuncts:
            d._deactivate_without_fixing_indicator()
    return ans