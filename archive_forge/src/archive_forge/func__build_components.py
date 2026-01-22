import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def _build_components(self, component_starts=None):
    """
        Each component is stored as a list of *entry points* to
        crossings.  The labeling of the entry points (equivalently
        oriented edges) is compatible with the DT convention that each
        crossing has both an odd and and even incoming strand.

        If provided the component_starts must consist of one
        CrossingEntryPoint per component. These need not satisfy the
        DT convention, but if they don't then they will be shifted by
        at most one along each component to ensure it.

        >>> L = Link('L10a90').mirror()
        >>> L.DT_code()
        [(20, 12, 18, 16), (8, 2, 4, 6, 14, 10)]
        >>> C0, C1 = L.link_components
        >>> L._build_components([C0[2], C1[0]])
        >>> L.DT_code()
        [(12, 18, 16, 20), (6, 8, 2, 4, 14, 10)]
        >>> L._build_components([C0[0], C1[2]])
        >>> L.DT_code()
        [(18, 10, 16, 14), (2, 4, 6, 12, 20, 8)]

        Here is one where we have to shift:

        >>> L._build_components([C0[0], C1[1]])
        >>> L.DT_code()
        [(18, 10, 16, 14), (2, 4, 6, 12, 20, 8)]
        >>> L._build_components([C0[-1], C1[0]])
        >>> L.DT_code()
        [(20, 12, 18, 16), (8, 2, 4, 6, 14, 10)]
        """
    if component_starts is not None:
        component_starts = [cs.crossing.entry_points()[cs.strand_index % 2] for cs in component_starts]
    remaining, components = (OrderedSet(self.crossing_entries()), LinkComponents())
    other_crossing_entries = []
    self.labels = labels = Labels()
    for c in self.crossings:
        c._clear_strand_info()
    while len(remaining):
        if component_starts:
            d = component_starts[len(components)]
        elif len(components) == 0:
            d = remaining.pop()
        else:
            found, comp_index = (False, 0)
            while not found and comp_index < len(components):
                others = other_crossing_entries[comp_index]
                if others:
                    for j, d in enumerate(others):
                        if d.component_label() is None:
                            if labels[d.other()] % 2 == 0:
                                d = d.next()
                            found = True
                            break
                    other_crossing_entries[comp_index] = others[j:]
                comp_index += 1
            if not found:
                d = remaining.pop()
        component = components.add(d)
        for c in component:
            labels.add(c)
        others = []
        for c in component:
            c.label_crossing(len(components) - 1, labels)
            o = c.other()
            if o.component_label() is None:
                others.append(o)
        other_crossing_entries.append(others)
        remaining.difference_update(component)
    if component_starts is not None and (not self._DT_convention_holds()):
        self._build_components(component_starts=None)
        new_starts = []
        for cep in component_starts:
            if cep.strand_label() % 2 == 0:
                new_starts.append(cep)
            else:
                new_starts.append(cep.next())
        self._build_components(component_starts=new_starts)
        assert self._DT_convention_holds()
    self.link_components = components