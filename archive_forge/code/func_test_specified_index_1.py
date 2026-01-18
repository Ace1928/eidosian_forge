import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_specified_index_1(self):
    """
        Components indexed by flattened sets and others
        """
    m = ConcreteModel()
    m.time = Set(initialize=[1, 2, 3])
    m.space = Set(initialize=[2, 4, 6])
    m.phase = Set(initialize=['p1', 'p2'])
    m.comp = Set(initialize=['a', 'b'])
    phase_comp = m.comp * m.phase
    n_phase_comp = len(m.phase) * len(m.comp)
    m.v = Var(m.time, m.comp, m.space, m.phase)

    @m.Block(m.time, m.comp, m.space, m.phase)
    def b(b, t, j, x, p):
        b.v1 = Var()
        if x != 2:
            b.v2 = Var()
    sets = (m.time, m.space)
    sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
    self.assertEqual(len(comps_list), len(sets_list))
    self.assertEqual(len(sets_list), 1)
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 2 and sets[0] is m.time and (sets[1] is m.space):
            self.assertEqual(len(comps), 2 * n_phase_comp)
            ref_data = set()
            ref_data.update((self._hashRef(Reference(m.v[:, j, :, p])) for j, p in phase_comp))
            ref_data.update((self._hashRef(Reference(m.b[:, j, :, p].v1)) for j, p in phase_comp))
            self.assertEqual(len(ref_data), len(comps))
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        else:
            raise RuntimeError()
    indices = ComponentMap([(m.space, 4)])
    sets_list, comps_list = flatten_components_along_sets(m, sets, Var, indices=indices)
    self.assertEqual(len(comps_list), len(sets_list))
    self.assertEqual(len(sets_list), 1)
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 2 and sets[0] is m.time and (sets[1] is m.space):
            self.assertEqual(len(comps), 3 * n_phase_comp)
            incomplete_slices = list((m.b[:, j, :, p].v2 for j, p in phase_comp))
            for ref in incomplete_slices:
                ref.attribute_errors_generate_exceptions = False
            incomplete_refs = list((Reference(sl) for sl in incomplete_slices))
            ref_data = set()
            ref_data.update((self._hashRef(Reference(m.v[:, j, :, p])) for j, p in phase_comp))
            ref_data.update((self._hashRef(Reference(m.b[:, j, :, p].v1)) for j, p in phase_comp))
            ref_data.update((self._hashRef(ref) for ref in incomplete_refs))
            self.assertEqual(len(ref_data), len(comps))
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        else:
            raise RuntimeError()
    indices = (3, 6)
    sets_list, comps_list = flatten_components_along_sets(m, sets, Var, indices=indices)
    self.assertEqual(len(comps_list), len(sets_list))
    self.assertEqual(len(sets_list), 1)
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 2 and sets[0] is m.time and (sets[1] is m.space):
            self.assertEqual(len(comps), 3 * n_phase_comp)
            incomplete_slices = list((m.b[:, j, :, p].v2 for j, p in phase_comp))
            for ref in incomplete_slices:
                ref.attribute_errors_generate_exceptions = False
            incomplete_refs = list((Reference(sl) for sl in incomplete_slices))
            ref_data = set()
            ref_data.update((self._hashRef(Reference(m.v[:, j, :, p])) for j, p in phase_comp))
            ref_data.update((self._hashRef(Reference(m.b[:, j, :, p].v1)) for j, p in phase_comp))
            ref_data.update((self._hashRef(ref) for ref in incomplete_refs))
            self.assertEqual(len(ref_data), len(comps))
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        else:
            raise RuntimeError()