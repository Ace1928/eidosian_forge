from typing import (
from cirq import protocols, value
from cirq.ops import global_phase_op, op_tree, raw_types
@value.value_equality(distinct_child_types=True)
class GateFamily:
    """Wrapper around gate instances/types describing a set of accepted gates.

    GateFamily supports initialization via

    - Non-parameterized instances of `cirq.Gate` (Instance Family).
    - Python types inheriting from `cirq.Gate` (Type Family).

    By default, the containment checks depend on the initialization type:

    - Instance Family: Containment check is done via `cirq.equal_up_to_global_phase`.
    - Type Family: Containment check is done by type comparison.

    For example:

    - Instance Family:

        >>> gate_family = cirq.GateFamily(cirq.X)
        >>> assert cirq.X in gate_family
        >>> assert cirq.Rx(rads=np.pi) in gate_family
        >>> assert cirq.X ** sympy.Symbol("theta") not in gate_family

    - Type Family:

        >>> gate_family = cirq.GateFamily(cirq.XPowGate)
        >>> assert cirq.X in gate_family
        >>> assert cirq.Rx(rads=np.pi) in gate_family
        >>> assert cirq.X ** sympy.Symbol("theta") in gate_family

    As seen in the examples above, GateFamily supports containment checks for instances of both
    `cirq.Operation` and `cirq.Gate`. By default, a `cirq.Operation` instance `op` is accepted if
    the underlying `op.gate` is accepted.

    Further constraints can be added on containment checks for `cirq.Operation` objects by setting
    `tags_to_accept` and/or `tags_to_ignore` in the GateFamily constructor. For a tagged
    operation, the underlying gate `op.gate` will be checked for containment only if both:

    - `op.tags` has no intersection with `tags_to_ignore`
    - `tags_to_accept` is not empty, then `op.tags` should have a non-empty intersection with
        `tags_to_accept`.

    If a `cirq.Operation` contains tags from both `tags_to_accept` and `tags_to_ignore`, it is
    rejected. Furthermore, tags cannot appear in both `tags_to_accept` and `tags_to_ignore`.

    For the purpose of tag comparisons, a `Gate` is considered as an `Operation` without tags.

    For example:

        >>> q = cirq.NamedQubit('q')
        >>> gate_family = cirq.GateFamily(cirq.ZPowGate, tags_to_accept=['accepted_tag'])
        >>> assert cirq.Z(q).with_tags('accepted_tag') in gate_family
        >>> assert cirq.Z(q).with_tags('other_tag') not in gate_family
        >>> assert cirq.Z(q) not in gate_family
        >>> assert cirq.Z not in gate_family
        ...
        >>> gate_family = cirq.GateFamily(cirq.ZPowGate, tags_to_ignore=['ignored_tag'])
        >>> assert cirq.Z(q).with_tags('ignored_tag') not in gate_family
        >>> assert cirq.Z(q).with_tags('other_tag') in gate_family
        >>> assert cirq.Z(q) in gate_family
        >>> assert cirq.Z in gate_family

    In order to create gate families with constraints on parameters of a gate
    type, users should derive from the `cirq.GateFamily` class and override the
    `_predicate` method used to check for gate containment.
    """

    def __init__(self, gate: Union[Type[raw_types.Gate], raw_types.Gate], *, name: Optional[str]=None, description: Optional[str]=None, ignore_global_phase: bool=True, tags_to_accept: Sequence[Hashable]=(), tags_to_ignore: Sequence[Hashable]=()) -> None:
        """Init GateFamily.

        Args:
            gate: A python `type` inheriting from `cirq.Gate` for type based membership checks, or
                a non-parameterized instance of a `cirq.Gate` for equality based membership checks.
            name: The name of the gate family.
            description: Human readable description of the gate family.
            ignore_global_phase: If True, value equality is checked via
                `cirq.equal_up_to_global_phase`.
            tags_to_accept: If non-empty, only `cirq.Operations` containing at least one tag in this
                sequence can be accepted.
            tags_to_ignore: Any `cirq.Operation` containing at least one tag in this sequence is
                rejected. Note that this takes precedence over `tags_to_accept`, so an operation
                which contains tags from both `tags_to_accept` and `tags_to_ignore` is rejected.

        Raises:
            ValueError: if `gate` is not a `cirq.Gate` instance or subclass.
            ValueError: if `gate` is a parameterized instance of `cirq.Gate`.
            ValueError: if `tags_to_accept` and `tags_to_ignore` contain common tags.
        """
        if not (isinstance(gate, raw_types.Gate) or (isinstance(gate, type) and issubclass(gate, raw_types.Gate))):
            raise ValueError(f'Gate {gate} must be an instance or subclass of `cirq.Gate`.')
        if isinstance(gate, raw_types.Gate) and protocols.is_parameterized(gate):
            raise ValueError(f'Gate {gate} must be a non-parameterized instance of `cirq.Gate`.')
        self._gate = gate
        self._tags_to_accept = frozenset(tags_to_accept)
        self._tags_to_ignore = frozenset(tags_to_ignore)
        self._name = name if name else self._default_name()
        self._description = description if description else self._default_description()
        self._ignore_global_phase = ignore_global_phase
        common_tags = self._tags_to_accept & self._tags_to_ignore
        if common_tags:
            raise ValueError(f"Tag(s) '{list(common_tags)}' cannot be in both tags_to_accept and tags_to_ignore.")

    def _gate_str(self, gettr: Callable[[Any], str]=str) -> str:
        return _gate_str(self.gate, gettr)

    def _gate_json(self) -> Union[raw_types.Gate, str]:
        return self.gate if not isinstance(self.gate, type) else protocols.json_cirq_type(self.gate)

    def _default_name(self) -> str:
        family_type = 'Instance' if isinstance(self.gate, raw_types.Gate) else 'Type'
        return f'{family_type} GateFamily: {self._gate_str()}'

    def _default_description(self) -> str:
        check_type = 'g == {}' if isinstance(self.gate, raw_types.Gate) else 'isinstance(g, {})'
        tags_to_accept_str = f'\nAccepted tags: {list(self._tags_to_accept)}' if self._tags_to_accept else ''
        tags_to_ignore_str = f'\nIgnored tags: {list(self._tags_to_ignore)}' if self._tags_to_ignore else ''
        return f'Accepts `cirq.Gate` instances `g` s.t. `{check_type.format(self._gate_str())}`' + tags_to_accept_str + tags_to_ignore_str

    @property
    def gate(self) -> Union[Type[raw_types.Gate], raw_types.Gate]:
        return self._gate

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def tags_to_accept(self) -> FrozenSet[Hashable]:
        return self._tags_to_accept

    @property
    def tags_to_ignore(self) -> FrozenSet[Hashable]:
        return self._tags_to_ignore

    def _predicate(self, gate: raw_types.Gate) -> bool:
        """Checks whether `cirq.Gate` instance `gate` belongs to this GateFamily.

        The default predicate depends on the gate family initialization type:

        - Instance Family: `cirq.equal_up_to_global_phase(gate, self.gate)`
            if self._ignore_global_phase else `gate == self.gate`.
        - Type Family: `isinstance(gate, self.gate)`.

        Args:
            gate: `cirq.Gate` instance which should be checked for containment.
        """
        if isinstance(self.gate, raw_types.Gate):
            return protocols.equal_up_to_global_phase(gate, self.gate) if self._ignore_global_phase else gate == self._gate
        return isinstance(gate, self.gate)

    def __contains__(self, item: Union[raw_types.Gate, raw_types.Operation]) -> bool:
        if self._tags_to_accept and (not isinstance(item, raw_types.Operation) or self._tags_to_accept.isdisjoint(item.tags)):
            return False
        if isinstance(item, raw_types.Operation) and (not self._tags_to_ignore.isdisjoint(item.tags)):
            return False
        if isinstance(item, raw_types.Operation):
            if item.gate is None:
                return False
            item = item.gate
        return self._predicate(item)

    def __str__(self) -> str:
        return f'{self.name}\n{self.description}'

    def __repr__(self) -> str:
        name_and_description = ''
        if self.name != self._default_name() or self.description != self._default_description():
            name_and_description = f'name="{self.name}", description="{self.description}", '
        return f'cirq.GateFamily(gate={self._gate_str(repr)}, {name_and_description}ignore_global_phase={self._ignore_global_phase}, tags_to_accept={self._tags_to_accept}, tags_to_ignore={self._tags_to_ignore})'

    def _value_equality_values_(self) -> Any:
        description = self.description if self.description != self._default_description() else None
        return (isinstance(self.gate, raw_types.Gate), self.gate, self.name, description, self._ignore_global_phase, self._tags_to_accept, self._tags_to_ignore)

    def _json_dict_(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {'gate': self._gate_json(), 'name': self.name, 'description': self.description, 'ignore_global_phase': self._ignore_global_phase}
        if self._tags_to_accept:
            d['tags_to_accept'] = list(self._tags_to_accept)
        if self._tags_to_ignore:
            d['tags_to_ignore'] = list(self._tags_to_ignore)
        return d

    @classmethod
    def _from_json_dict_(cls, gate, name, description, ignore_global_phase, tags_to_accept=(), tags_to_ignore=(), **kwargs) -> 'GateFamily':
        if isinstance(gate, str):
            gate = protocols.cirq_type_from_json(gate)
        return cls(gate, name=name, description=description, ignore_global_phase=ignore_global_phase, tags_to_accept=tags_to_accept, tags_to_ignore=tags_to_ignore)