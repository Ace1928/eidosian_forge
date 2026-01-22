import operator
import typing
from collections.abc import MappingView, MutableMapping, MutableSet
class ParameterTable(MutableMapping):
    """Class for tracking references to circuit parameters by specific
    instruction instances.

    Keys are parameters. Values are of type :class:`~ParameterReferences`,
    which overrides membership testing to be referential for instructions,
    and is set-like. Elements of :class:`~ParameterReferences`
    are tuples of ``(instruction, param_index)``.
    """
    __slots__ = ['_table', '_keys', '_names']

    class _GlobalPhaseSentinel:
        __slots__ = ()

        def __copy__(self):
            return self

        def __deepcopy__(self, memo=None):
            return self

        def __reduce__(self):
            return (operator.attrgetter('GLOBAL_PHASE'), (ParameterTable,))

        def __repr__(self):
            return '<global-phase sentinel>'
    GLOBAL_PHASE = _GlobalPhaseSentinel()
    'Tracking object to indicate that a reference refers to the global phase of a circuit.'

    def __init__(self, mapping=None):
        """Create a new instance, initialized with ``mapping`` if provided.

        Args:
            mapping (Mapping[Parameter, ParameterReferences]):
                Mapping of parameter to the set of parameter slots that reference
                it.

        Raises:
            ValueError: A value in ``mapping`` is not a :class:`~ParameterReferences`.
        """
        if mapping is not None:
            if any((not isinstance(refs, ParameterReferences) for refs in mapping.values())):
                raise ValueError('Values must be of type ParameterReferences')
            self._table = mapping.copy()
        else:
            self._table = {}
        self._keys = set(self._table)
        self._names = {x.name: x for x in self._table}

    def __getitem__(self, key):
        return self._table[key]

    def __setitem__(self, parameter, refs):
        """Associate a parameter with the set of parameter slots ``(instruction, param_index)``
        that reference it.

        .. note::

            Items in ``refs`` are considered unique if their ``instruction`` is referentially
            unique. See :class:`~ParameterReferences` for details.

        Args:
            parameter (Parameter): the parameter
            refs (Union[ParameterReferences, Iterable[(Instruction, int)]]): the parameter slots.
                If this is an iterable, a new :class:`~ParameterReferences` is created from its
                contents.
        """
        if not isinstance(refs, ParameterReferences):
            refs = ParameterReferences(refs)
        self._table[parameter] = refs
        self._keys.add(parameter)
        self._names[parameter.name] = parameter

    def get_keys(self):
        """Return a set of all keys in the parameter table

        Returns:
            set: A set of all the keys in the parameter table
        """
        return self._keys

    def get_names(self):
        """Return a set of all parameter names in the parameter table

        Returns:
            set: A set of all the names in the parameter table
        """
        return self._names.keys()

    def parameter_from_name(self, name: str, default: typing.Any=None):
        """Get a :class:`.Parameter` with references in this table by its string name.

        If the parameter is not present, return the ``default`` value.

        Args:
            name: The name of the :class:`.Parameter`
            default: The object that should be returned if the parameter is missing.
        """
        return self._names.get(name, default)

    def discard_references(self, expression, key):
        """Remove all references to parameters contained within ``expression`` at the given table
        ``key``.  This also discards parameter entries from the table if they have no further
        references.  No action is taken if the object is not tracked."""
        for parameter in expression.parameters:
            if (refs := self._table.get(parameter)) is not None:
                if len(refs) == 1:
                    del self[parameter]
                else:
                    refs.discard(key)

    def __delitem__(self, key):
        del self._table[key]
        self._keys.discard(key)
        del self._names[key.name]

    def __iter__(self):
        return iter(self._table)

    def __len__(self):
        return len(self._table)

    def __repr__(self):
        return f'ParameterTable({repr(self._table)})'