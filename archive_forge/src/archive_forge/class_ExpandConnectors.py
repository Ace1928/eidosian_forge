import logging
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.log import is_debug_set
import pyomo.core.expr as EXPR
from pyomo.core.base import (
from pyomo.core.base.connector import _ConnectorData, ScalarConnector
@TransformationFactory.register('core.expand_connectors', doc='Expand all connectors in the model to simple constraints')
class ExpandConnectors(Transformation):

    def _apply_to(self, instance, **kwds):
        if is_debug_set(logger):
            logger.debug('Calling ConnectorExpander')
        connectorsFound = False
        for c in instance.component_data_objects(Connector):
            connectorsFound = True
            break
        if not connectorsFound:
            return
        if is_debug_set(logger):
            logger.debug('   Connectors found!')
        connector_list = []
        constraint_list = []
        groupID = 0
        connector_groups = dict()
        matched_connectors = ComponentMap()
        found = ComponentSet()
        connector_types = set([ScalarConnector, _ConnectorData])
        for constraint in instance.component_data_objects(Constraint, sort=SortComponents.deterministic):
            ref = None
            for c in EXPR.identify_components(constraint.body, connector_types):
                found.add(c)
                if c in matched_connectors:
                    if ref is None:
                        ref = matched_connectors[c]
                    elif ref is not matched_connectors[c]:
                        src = matched_connectors[c]
                        if len(ref) < len(src):
                            ref, src = (src, ref)
                        ref.update(src)
                        for _ in src:
                            matched_connectors[_] = ref
                        del connector_groups[id(src)]
                else:
                    connector_list.append(c)
                    if ref is None:
                        ref = ComponentSet()
                        connector_groups[id(ref)] = (groupID, ref)
                        groupID += 1
                    ref.add(c)
                    matched_connectors[c] = ref
            if ref is not None:
                constraint_list.append((constraint, found))
                found = ComponentSet()
        known_conn_sets = {}
        for groupID, conn_set in sorted(connector_groups.values()):
            known_conn_sets[id(conn_set)] = self._validate_and_expand_connector_set(conn_set)
        for constraint, conn_set in constraint_list:
            cList = ConstraintList()
            constraint.parent_block().add_component('%s.expanded' % (constraint.getname(fully_qualified=False),), cList)
            connId = next(iter(conn_set))
            ref = known_conn_sets[id(matched_connectors[connId])]
            for k, v in sorted(ref.items()):
                if v[1] >= 0:
                    _iter = v[0]
                else:
                    _iter = (v[0],)
                for idx in _iter:
                    substitution = {}
                    for c in conn_set:
                        if v[1] >= 0:
                            new_v = c.vars[k][idx]
                        elif k in c.aggregators:
                            new_v = c.vars[k].add()
                        else:
                            new_v = c.vars[k]
                        substitution[id(c)] = new_v
                    cList.add((constraint.lower, EXPR.clone_expression(constraint.body, substitution), constraint.upper))
            constraint.deactivate()
        for conn in connector_list:
            block = conn.parent_block()
            for var, aggregator in conn.aggregators.items():
                c = Constraint(expr=aggregator(block, conn.vars[var]))
                block.add_component('%s.%s.aggregate' % (conn.getname(fully_qualified=True), var), c)

    def _validate_and_expand_connector_set(self, connectors):
        ref = {}
        for c in connectors:
            for k, v in c.vars.items():
                if k in ref:
                    continue
                if v is None:
                    continue
                _len = -2 if k in c.aggregators else -1 if not hasattr(v, 'is_indexed') or not v.is_indexed() else len(v)
                ref[k] = (v, _len, c)
        if not ref:
            logger.warning('Cannot identify a reference connector: no connectors in the connector set have assigned variables:\n\t(%s)' % (', '.join(sorted((c.name for c in connectors))),))
            return ref
        empty_or_partial = []
        for c in connectors:
            c_is_partial = False
            if not c.vars:
                empty_or_partial.append(c)
                continue
            for k, v in ref.items():
                if k not in c.vars:
                    raise ValueError("Connector mismatch: Connector '%s' missing variable '%s' (appearing in reference connector '%s')" % (c.name, k, v[2].name))
                _v = c.vars[k]
                if _v is None:
                    if not c_is_partial:
                        empty_or_partial.append(c)
                        c_is_partial = True
                    continue
                _len = -3 if _v is None else -2 if k in c.aggregators else -1 if not hasattr(_v, 'is_indexed') or not _v.is_indexed() else len(_v)
                if (_len >= 0) ^ (v[1] >= 0):
                    raise ValueError("Connector mismatch: Connector variable '%s' mixing indexed and non-indexed targets on connectors '%s' and '%s'" % (k, v[2].name, c.name))
                if _len >= 0 and _len != v[1]:
                    raise ValueError("Connector mismatch: Connector variable '%s' index mismatch (%s elements in reference connector '%s', but %s elements in connector '%s')" % (k, v[1], v[2].name, _len, c.name))
                if v[1] >= 0 and len(v[0].index_set() ^ _v.index_set()):
                    raise ValueError("Connector mismatch: Connector variable '%s' has mismatched indices on connectors '%s' and '%s'" % (k, v[2].name, c.name))
        sorted_refs = sorted(ref.items())
        if len(empty_or_partial) > 1:
            empty_or_partial.sort(key=lambda x: x.getname(fully_qualified=True))
        for c in empty_or_partial:
            block = c.parent_block()
            for k, v in sorted_refs:
                if k in c.vars and c.vars[k] is not None:
                    continue
                if v[1] >= 0:
                    idx = (v[0].index_set(),)
                else:
                    idx = ()
                var_args = {}
                try:
                    var_args['domain'] = v[0].domain
                except AttributeError:
                    pass
                try:
                    var_args['bounds'] = v[0].bounds
                except AttributeError:
                    pass
                new_var = Var(*idx, **var_args)
                block.add_component('%s.auto.%s' % (c.getname(fully_qualified=True), k), new_var)
                if idx:
                    for i in idx[0]:
                        new_var[i].domain = v[0][i].domain
                        new_var[i].setlb(v[0][i].lb)
                        new_var[i].setub(v[0][i].ub)
                c.vars[k] = new_var
        return ref