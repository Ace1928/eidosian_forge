from __future__ import annotations
import abc
import copy
import functools
import itertools
import multiprocessing as mp
import sys
import warnings
from collections.abc import Callable, Iterable
from typing import List, Tuple, Union, Dict, Any
import numpy as np
import rustworkx as rx
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import PulseError, UnassignedReferenceError
from qiskit.pulse.instructions import Instruction, Reference
from qiskit.pulse.utils import instruction_duration_validation
from qiskit.pulse.reference_manager import ReferenceManager
from qiskit.utils.multiprocessing import is_main_process
class ScheduleBlock:
    """Time-ordered sequence of instructions with alignment context.

    :class:`.ScheduleBlock` supports lazy scheduling of context instructions,
    i.e. their timeslots is always generated at runtime.
    This indicates we can parametrize instruction durations as well as
    other parameters. In contrast to :class:`.Schedule` being somewhat static,
    :class:`.ScheduleBlock` is a dynamic representation of a pulse program.

    .. rubric:: Pulse Builder

    The Qiskit pulse builder is a domain specific language that is developed on top of
    the schedule block. Use of the builder syntax will improve the workflow of
    pulse programming. See :ref:`pulse_builder` for a user guide.

    .. rubric:: Alignment contexts

    A schedule block is always relatively scheduled.
    Instead of taking individual instructions with absolute execution time ``t0``,
    the schedule block defines a context of scheduling and instructions
    under the same context are scheduled in the same manner (alignment).
    Several contexts are available in :ref:`pulse_alignments`.
    A schedule block is instantiated with one of these alignment contexts.
    The default context is :class:`AlignLeft`, for which all instructions are left-justified,
    in other words, meaning they use as-soon-as-possible scheduling.

    If you need an absolute-time interval in between instructions, you can explicitly
    insert :class:`~qiskit.pulse.instructions.Delay` instructions.

    .. rubric:: Nested blocks

    A schedule block can contain other nested blocks with different alignment contexts.
    This enables advanced scheduling, where a subset of instructions is
    locally scheduled in a different manner.
    Note that a :class:`.Schedule` instance cannot be directly added to a schedule block.
    To add a :class:`.Schedule` instance, wrap it in a :class:`.Call` instruction.
    This is implicitly performed when a schedule is added through the :ref:`pulse_builder`.

    .. rubric:: Unsupported operations

    Because the schedule block representation lacks timeslots, it cannot
    perform particular :class:`.Schedule` operations such as :meth:`insert` or :meth:`shift` that
    require instruction start time ``t0``.
    In addition, :meth:`exclude` and :meth:`filter` methods are not supported
    because these operations may identify the target instruction with ``t0``.
    Except for these operations, :class:`.ScheduleBlock` provides full compatibility
    with :class:`.Schedule`.

    .. rubric:: Subroutine

    The timeslots-free representation offers much greater flexibility for writing pulse programs.
    Because :class:`.ScheduleBlock` only cares about the ordering of the child blocks
    we can add an undefined pulse sequence as a subroutine of the main program.
    If your program contains the same sequence multiple times, this representation may
    reduce the memory footprint required by the program construction.
    Such a subroutine is realized by the special compiler directive
    :class:`~qiskit.pulse.instructions.Reference` that is defined by
    a unique set of reference key strings to the subroutine.
    The (executable) subroutine is separately stored in the main program.
    Appended reference directives are resolved when the main program is executed.
    Subroutines must be assigned through :meth:`assign_references` before execution.

    One way to reference a subroutine in a schedule is to use the pulse
    builder's :func:`~qiskit.pulse.builder.reference`  function to declare an
    unassigned reference.  In this example, the program is called with the
    reference key "grand_child".  You can call a subroutine without specifying
    a substantial program.

    .. code-block::

        from qiskit import pulse
        from qiskit.circuit.parameter import Parameter

        amp1 = Parameter("amp1")
        amp2 = Parameter("amp2")

        with pulse.build() as sched_inner:
            pulse.play(pulse.Constant(100, amp1), pulse.DriveChannel(0))

        with pulse.build() as sched_outer:
            with pulse.align_right():
                pulse.reference("grand_child")
                pulse.play(pulse.Constant(200, amp2), pulse.DriveChannel(0))

    Now you assign the inner pulse program to this reference.

    .. code-block::

        sched_outer.assign_references({("grand_child", ): sched_inner})
        print(sched_outer.parameters)

    .. parsed-literal::

       {Parameter(amp1), Parameter(amp2)}

    The outer program now has the parameter ``amp2`` from the inner program,
    indicating that the inner program's data has been made available to the
    outer program.
    The program calling the "grand_child" has a reference program description
    which is accessed through :attr:`ScheduleBlock.references`.

    .. code-block::

        print(sched_outer.references)

    .. parsed-literal::

       ReferenceManager:
         - ('grand_child',): ScheduleBlock(Play(Constant(duration=100, amp=amp1,...

    Finally, you may want to call this program from another program.
    Here we try a different approach to define subroutine. Namely, we call
    a subroutine from the root program with the actual program ``sched2``.

    .. code-block::

        amp3 = Parameter("amp3")

        with pulse.build() as main:
            pulse.play(pulse.Constant(300, amp3), pulse.DriveChannel(0))
            pulse.call(sched_outer, name="child")

        print(main.parameters)

    .. parsed-literal::

       {Parameter(amp1), Parameter(amp2), Parameter(amp3}

    This implicitly creates a reference named "child" within
    the root program and assigns ``sched_outer`` to it.

    Note that the root program is only aware of its direct references.

    .. code-block::

        print(main.references)

    .. parsed-literal::

       ReferenceManager:
         - ('child',): ScheduleBlock(ScheduleBlock(ScheduleBlock(Play(Con...

    As you can see the main program cannot directly assign a subroutine to the "grand_child" because
    this subroutine is not called within the root program, i.e. it is indirectly called by "child".
    However, the returned :class:`.ReferenceManager` is a dict-like object, and you can still
    reach to "grand_child" via the "child" program with the following chained dict access.

    .. code-block::

        main.references[("child", )].references[("grand_child", )]

    Note that :attr:`ScheduleBlock.parameters` still collects all parameters
    also from the subroutine once it's assigned.
    """
    __slots__ = ('_parent', '_name', '_reference_manager', '_parameter_manager', '_alignment_context', '_blocks', '_metadata')
    prefix = 'block'
    instances_counter = itertools.count()

    def __init__(self, name: str | None=None, metadata: dict | None=None, alignment_context=None):
        """Create an empty schedule block.

        Args:
            name: Name of this schedule. Defaults to an autogenerated string if not provided.
            metadata: Arbitrary key value metadata to associate with the schedule. This gets
                stored as free-form data in a dict in the
                :attr:`~qiskit.pulse.ScheduleBlock.metadata` attribute. It will not be directly
                used in the schedule.
            alignment_context (AlignmentKind): ``AlignmentKind`` instance that manages
                scheduling of instructions in this block.

        Raises:
            TypeError: if metadata is not a dict.
        """
        from qiskit.pulse.parameter_manager import ParameterManager
        from qiskit.pulse.transforms import AlignLeft
        if name is None:
            name = self.prefix + str(next(self.instances_counter))
            if sys.platform != 'win32' and (not is_main_process()):
                name += f'-{mp.current_process().pid}'
        self._parent: ScheduleBlock | None = None
        self._name = name
        self._parameter_manager = ParameterManager()
        self._reference_manager = ReferenceManager()
        self._alignment_context = alignment_context or AlignLeft()
        self._blocks: list['BlockComponent'] = []
        self._parameter_manager.update_parameter_table(self._alignment_context)
        if not isinstance(metadata, dict) and metadata is not None:
            raise TypeError('Only a dictionary or None is accepted for schedule metadata')
        self._metadata = metadata or {}

    @classmethod
    def initialize_from(cls, other_program: Any, name: str | None=None) -> 'ScheduleBlock':
        """Create new schedule object with metadata of another schedule object.

        Args:
            other_program: Qiskit program that provides metadata to new object.
            name: Name of new schedule. Name of ``block`` is used by default.

        Returns:
            New block object with name and metadata.

        Raises:
            PulseError: When ``other_program`` does not provide necessary information.
        """
        try:
            name = name or other_program.name
            if other_program.metadata:
                metadata = other_program.metadata.copy()
            else:
                metadata = None
            try:
                alignment_context = other_program.alignment_context
            except AttributeError:
                alignment_context = None
            return cls(name=name, metadata=metadata, alignment_context=alignment_context)
        except AttributeError as ex:
            raise PulseError(f'{cls.__name__} cannot be initialized from the program data {other_program.__class__.__name__}.') from ex

    @property
    def name(self) -> str:
        """Return name of this schedule"""
        return self._name

    @property
    def metadata(self) -> dict[str, Any]:
        """The user provided metadata associated with the schedule.

        User provided ``dict`` of metadata for the schedule.
        The metadata contents do not affect the semantics of the program
        but are used to influence the execution of the schedule. It is expected
        to be passed between all transforms of the schedule and that providers
        will associate any schedule metadata with the results it returns from the
        execution of that schedule.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        """Update the schedule metadata"""
        if not isinstance(metadata, dict) and metadata is not None:
            raise TypeError('Only a dictionary or None is accepted for schedule metadata')
        self._metadata = metadata or {}

    @property
    def alignment_context(self):
        """Return alignment instance that allocates block component to generate schedule."""
        return self._alignment_context

    def is_schedulable(self) -> bool:
        """Return ``True`` if all durations are assigned."""
        for context_param in self._alignment_context._context_params:
            if isinstance(context_param, ParameterExpression):
                return False
        for elm in self.blocks:
            if isinstance(elm, ScheduleBlock):
                if not elm.is_schedulable():
                    return False
            else:
                try:
                    if not isinstance(elm.duration, int):
                        return False
                except UnassignedReferenceError:
                    return False
        return True

    @property
    @_require_schedule_conversion
    def duration(self) -> int:
        """Duration of this schedule block."""
        return self.duration

    @property
    def channels(self) -> tuple[Channel, ...]:
        """Returns channels that this schedule block uses."""
        chans: set[Channel] = set()
        for elm in self.blocks:
            if isinstance(elm, Reference):
                raise UnassignedReferenceError(f'This schedule contains unassigned reference {elm.ref_keys} and channels are ambiguous. Please assign the subroutine first.')
            chans = chans | set(elm.channels)
        return tuple(chans)

    @property
    @_require_schedule_conversion
    def instructions(self) -> tuple[tuple[int, Instruction]]:
        """Get the time-ordered instructions from self."""
        return self.instructions

    @property
    def blocks(self) -> tuple['BlockComponent', ...]:
        """Get the block elements added to self.

        .. note::

            The sequence of elements is returned in order of addition. Because the first element is
            schedule first, e.g. FIFO, the returned sequence is roughly time-ordered.
            However, in the parallel alignment context, especially in
            the as-late-as-possible scheduling, or :class:`.AlignRight` context,
            the actual timing of when the instructions are issued is unknown until
            the :class:`.ScheduleBlock` is scheduled and converted into a :class:`.Schedule`.
        """
        blocks = []
        for elm in self._blocks:
            if isinstance(elm, Reference):
                elm = self.references.get(elm.ref_keys, None) or elm
            blocks.append(elm)
        return tuple(blocks)

    @property
    def parameters(self) -> set[Parameter]:
        """Return unassigned parameters with raw names."""
        out_params = set()
        out_params |= self._parameter_manager.parameters
        for subroutine in self.references.values():
            if subroutine is None:
                continue
            out_params |= subroutine.parameters
        return out_params

    @property
    def references(self) -> ReferenceManager:
        """Return a reference manager of the current scope."""
        if self._parent is not None:
            return self._parent.references
        return self._reference_manager

    @_require_schedule_conversion
    def ch_duration(self, *channels: Channel) -> int:
        """Return the time of the end of the last instruction over the supplied channels.

        Args:
            *channels: Channels within ``self`` to include.
        """
        return self.ch_duration(*channels)

    def append(self, block: 'BlockComponent', name: str | None=None, inplace: bool=True) -> 'ScheduleBlock':
        """Return a new schedule block with ``block`` appended to the context block.
        The execution time is automatically assigned when the block is converted into schedule.

        Args:
            block: ScheduleBlock to be appended.
            name: Name of the new ``Schedule``. Defaults to name of ``self``.
            inplace: Perform operation inplace on this schedule. Otherwise,
                return a new ``Schedule``.

        Returns:
            Schedule block with appended schedule.

        Raises:
            PulseError: When invalid schedule type is specified.
        """
        if not isinstance(block, (ScheduleBlock, Instruction)):
            raise PulseError(f'Appended `schedule` {block.__class__.__name__} is invalid type. Only `Instruction` and `ScheduleBlock` can be accepted.')
        if not inplace:
            schedule = copy.deepcopy(self)
            schedule._name = name or self.name
            schedule.append(block, inplace=True)
            return schedule
        if isinstance(block, Reference) and block.ref_keys not in self.references:
            self.references[block.ref_keys] = None
        elif isinstance(block, ScheduleBlock):
            block = copy.deepcopy(block)
            if block.is_referenced():
                if block._parent is not None:
                    for ref in _get_references(block._blocks):
                        self.references[ref.ref_keys] = block.references[ref.ref_keys]
                else:
                    for ref_keys, ref in block._reference_manager.items():
                        self.references[ref_keys] = ref
                    block._reference_manager.clear()
            block._parent = self
        self._blocks.append(block)
        self._parameter_manager.update_parameter_table(block)
        return self

    def filter(self, *filter_funcs: Callable[..., bool], channels: Iterable[Channel] | None=None, instruction_types: Iterable[abc.ABCMeta] | abc.ABCMeta=None, check_subroutine: bool=True):
        """Return a new ``ScheduleBlock`` with only the instructions from this ``ScheduleBlock``
        which pass though the provided filters; i.e. an instruction will be retained if
        every function in ``filter_funcs`` returns ``True``, the instruction occurs on
        a channel type contained in ``channels``, and the instruction type is contained
        in ``instruction_types``.

        .. warning::
            Because ``ScheduleBlock`` is not aware of the execution time of
            the context instructions, filtering out some instructions may
            change the execution time of the remaining instructions.

        If no arguments are provided, ``self`` is returned.

        Args:
            filter_funcs: A list of Callables which take a ``Instruction`` and return a bool.
            channels: For example, ``[DriveChannel(0), AcquireChannel(0)]``.
            instruction_types: For example, ``[PulseInstruction, AcquireInstruction]``.
            check_subroutine: Set `True` to individually filter instructions inside a subroutine
                defined by the :py:class:`~qiskit.pulse.instructions.Call` instruction.

        Returns:
            ``ScheduleBlock`` consisting of instructions that matches with filtering condition.
        """
        from qiskit.pulse.filters import composite_filter, filter_instructions
        filters = composite_filter(channels, instruction_types)
        filters.extend(filter_funcs)
        return filter_instructions(self, filters=filters, negate=False, recurse_subroutines=check_subroutine)

    def exclude(self, *filter_funcs: Callable[..., bool], channels: Iterable[Channel] | None=None, instruction_types: Iterable[abc.ABCMeta] | abc.ABCMeta=None, check_subroutine: bool=True):
        """Return a new ``ScheduleBlock`` with only the instructions from this ``ScheduleBlock``
        *failing* at least one of the provided filters.
        This method is the complement of py:meth:`~self.filter`, so that::

            self.filter(args) + self.exclude(args) == self in terms of instructions included.

        .. warning::
            Because ``ScheduleBlock`` is not aware of the execution time of
            the context instructions, excluding some instructions may
            change the execution time of the remaining instructions.

        Args:
            filter_funcs: A list of Callables which take a ``Instruction`` and return a bool.
            channels: For example, ``[DriveChannel(0), AcquireChannel(0)]``.
            instruction_types: For example, ``[PulseInstruction, AcquireInstruction]``.
            check_subroutine: Set `True` to individually filter instructions inside of a subroutine
                defined by the :py:class:`~qiskit.pulse.instructions.Call` instruction.

        Returns:
            ``ScheduleBlock`` consisting of instructions that do not match with
            at least one of filtering conditions.
        """
        from qiskit.pulse.filters import composite_filter, filter_instructions
        filters = composite_filter(channels, instruction_types)
        filters.extend(filter_funcs)
        return filter_instructions(self, filters=filters, negate=True, recurse_subroutines=check_subroutine)

    def replace(self, old: 'BlockComponent', new: 'BlockComponent', inplace: bool=True) -> 'ScheduleBlock':
        """Return a ``ScheduleBlock`` with the ``old`` component replaced with a ``new``
        component.

        Args:
            old: Schedule block component to replace.
            new: Schedule block component to replace with.
            inplace: Replace instruction by mutably modifying this ``ScheduleBlock``.

        Returns:
            The modified schedule block with ``old`` replaced by ``new``.
        """
        if not inplace:
            schedule = copy.deepcopy(self)
            return schedule.replace(old, new, inplace=True)
        if old not in self._blocks:
            return self
        all_references = ReferenceManager()
        if isinstance(new, ScheduleBlock):
            new = copy.deepcopy(new)
            all_references.update(new.references)
            new._reference_manager.clear()
            new._parent = self
        for ref_key, subroutine in self.references.items():
            if ref_key in all_references:
                warnings.warn(f'Reference {ref_key} conflicts with substituted program {new.name}. Existing reference has been replaced with new reference.', UserWarning)
                continue
            all_references[ref_key] = subroutine
        self._parameter_manager.clear()
        self._parameter_manager.update_parameter_table(self._alignment_context)
        new_elms = []
        for elm in self._blocks:
            if elm == old:
                elm = new
            self._parameter_manager.update_parameter_table(elm)
            new_elms.append(elm)
        self._blocks = new_elms
        self.references.clear()
        root = self
        while root._parent is not None:
            root = root._parent
        for ref in _get_references(root._blocks):
            self.references[ref.ref_keys] = all_references[ref.ref_keys]
        return self

    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        return any(self.parameters)

    def is_referenced(self) -> bool:
        """Return True iff the current schedule block contains reference to subroutine."""
        return len(self.references) > 0

    def assign_parameters(self, value_dict: dict[ParameterExpression, ParameterValueType], inplace: bool=True) -> 'ScheduleBlock':
        """Assign the parameters in this schedule according to the input.

        Args:
            value_dict: A mapping from Parameters to either numeric values or another
                Parameter expression.
            inplace: Set ``True`` to override this instance with new parameter.

        Returns:
            Schedule with updated parameters.

        Raises:
            PulseError: When the block is nested into another block.
        """
        if not inplace:
            new_schedule = copy.deepcopy(self)
            return new_schedule.assign_parameters(value_dict, inplace=True)
        self._parameter_manager.assign_parameters(pulse_program=self, value_dict=value_dict)
        for subroutine in self._reference_manager.values():
            if subroutine is None:
                continue
            subroutine.assign_parameters(value_dict=value_dict, inplace=True)
        return self

    def assign_references(self, subroutine_dict: dict[str | tuple[str, ...], 'ScheduleBlock'], inplace: bool=True) -> 'ScheduleBlock':
        """Assign schedules to references.

        It is only capable of assigning a schedule block to immediate references
        which are directly referred within the current scope.
        Let's see following example:

        .. code-block:: python

            from qiskit import pulse

            with pulse.build() as subroutine:
                pulse.delay(10, pulse.DriveChannel(0))

            with pulse.build() as sub_prog:
                pulse.reference("A")

            with pulse.build() as main_prog:
                pulse.reference("B")

        In above example, the ``main_prog`` can refer to the subroutine "root::B" and the
        reference of "B" to program "A", i.e., "B::A", is not defined in the root namespace.
        This prevents breaking the reference "root::B::A" by the assignment of "root::B".
        For example, if a user could indirectly assign "root::B::A" from the root program,
        one can later assign another program to "root::B" that doesn't contain "A" within it.
        In this situation, a reference "root::B::A" would still live in
        the reference manager of the root.
        However, the subroutine "root::B::A" would no longer be used in the actual pulse program.
        To assign subroutine "A" to ``nested_prog`` as a nested subprogram of ``main_prog``,
        you must first assign "A" of the ``sub_prog``,
        and then assign the ``sub_prog`` to the ``main_prog``.

        .. code-block:: python

            sub_prog.assign_references({("A", ): nested_prog}, inplace=True)
            main_prog.assign_references({("B", ): sub_prog}, inplace=True)

        Alternatively, you can also write

        .. code-block:: python

            main_prog.assign_references({("B", ): sub_prog}, inplace=True)
            main_prog.references[("B", )].assign_references({"A": nested_prog}, inplace=True)

        Here :attr:`.references` returns a dict-like object, and you can
        mutably update the nested reference of the particular subroutine.

        .. note::

            Assigned programs are deep-copied to prevent an unexpected update.

        Args:
            subroutine_dict: A mapping from reference key to schedule block of the subroutine.
            inplace: Set ``True`` to override this instance with new subroutine.

        Returns:
            Schedule block with assigned subroutine.

        Raises:
            PulseError: When reference key is not defined in the current scope.
        """
        if not inplace:
            new_schedule = copy.deepcopy(self)
            return new_schedule.assign_references(subroutine_dict, inplace=True)
        for key, subroutine in subroutine_dict.items():
            if key not in self.references:
                unassigned_keys = ', '.join(map(repr, self.references.unassigned()))
                raise PulseError(f"Reference instruction with {key} doesn't exist in the current scope: {unassigned_keys}")
            self.references[key] = copy.deepcopy(subroutine)
        return self

    def get_parameters(self, parameter_name: str) -> list[Parameter]:
        """Get parameter object bound to this schedule by string name.

        Note that we can define different parameter objects with the same name,
        because these different objects are identified by their unique uuid.
        For example,

        .. code-block:: python

            from qiskit import pulse, circuit

            amp1 = circuit.Parameter("amp")
            amp2 = circuit.Parameter("amp")

            with pulse.build() as sub_prog:
                pulse.play(pulse.Constant(100, amp1), pulse.DriveChannel(0))

            with pulse.build() as main_prog:
                pulse.call(sub_prog, name="sub")
                pulse.play(pulse.Constant(100, amp2), pulse.DriveChannel(0))

            main_prog.get_parameters("amp")

        This returns a list of two parameters ``amp1`` and ``amp2``.

        Args:
            parameter_name: Name of parameter.

        Returns:
            Parameter objects that have corresponding name.
        """
        matched = [p for p in self.parameters if p.name == parameter_name]
        return matched

    def __len__(self) -> int:
        """Return number of instructions in the schedule."""
        return len(self.blocks)

    def __eq__(self, other: object) -> bool:
        """Test if two ScheduleBlocks are equal.

        Equality is checked by verifying there is an equal instruction at every time
        in ``other`` for every instruction in this ``ScheduleBlock``. This check is
        performed by converting the instruction representation into directed acyclic graph,
        in which execution order of every instruction is evaluated correctly across all channels.
        Also ``self`` and ``other`` should have the same alignment context.

        .. warning::

            This does not check for logical equivalency. Ie.,

            ```python
            >>> Delay(10, DriveChannel(0)) + Delay(10, DriveChannel(0))
                == Delay(20, DriveChannel(0))
            False
            ```
        """
        if not isinstance(other, type(self)):
            return False
        if self.alignment_context != other.alignment_context:
            return False
        if len(self) != len(other):
            return False
        from qiskit.pulse.transforms.dag import block_to_dag as dag
        if not rx.is_isomorphic_node_match(dag(self), dag(other), lambda x, y: x == y):
            return False
        return True

    def __repr__(self) -> str:
        name = format(self._name) if self._name else ''
        blocks = ', '.join([repr(instr) for instr in self.blocks[:50]])
        if len(self.blocks) > 25:
            blocks += ', ...'
        return '{}({}, name="{}", transform={})'.format(self.__class__.__name__, blocks, name, repr(self.alignment_context))

    def __add__(self, other: 'BlockComponent') -> 'ScheduleBlock':
        """Return a new schedule with ``other`` inserted within ``self`` at ``start_time``."""
        return self.append(other)