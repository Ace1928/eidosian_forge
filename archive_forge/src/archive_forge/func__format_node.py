import functools
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow import states
from taskflow.types import tree
from taskflow.utils import misc
def _format_node(self, storage, cache, node):
    """Formats a single tree node into a string version."""
    if node.metadata['kind'] == compiler.FLOW:
        flow = node.item
        flow_name = flow.name
        return "Flow '%s'" % flow_name
    elif node.metadata['kind'] in compiler.ATOMS:
        atom = node.item
        atom_name = atom.name
        atom_attrs = {}
        intention, intention_found = _cached_get(cache, 'intentions', atom_name, storage.get_atom_intention, atom_name)
        if intention_found:
            atom_attrs['intention'] = intention
        state, state_found = _cached_get(cache, 'states', atom_name, storage.get_atom_state, atom_name)
        if state_found:
            atom_attrs['state'] = state
        if atom_name not in self._hide_inputs_outputs_of:
            fetch_mapped_args = functools.partial(storage.fetch_mapped_args, atom.rebind, atom_name=atom_name, optional_args=atom.optional)
            requires, requires_found = _cached_get(cache, 'requires', atom_name, fetch_mapped_args)
            if requires_found:
                atom_attrs['requires'] = requires
            provides, provides_found = _cached_get(cache, 'provides', atom_name, storage.get_execute_result, atom_name)
            if provides_found:
                atom_attrs['provides'] = provides
        if atom_attrs:
            return "Atom '%s' %s" % (atom_name, atom_attrs)
        else:
            return "Atom '%s'" % atom_name
    else:
        raise TypeError("Unable to format node, unknown node kind '%s' encountered" % node.metadata['kind'])