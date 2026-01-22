from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING
from qiskit.pulse import channels, exceptions, instructions, utils
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.schedule import Schedule
from qiskit.providers.backend import BackendV2
def _measure_v1(qubits: Sequence[int], inst_map: InstructionScheduleMap, meas_map: list[list[int]] | dict[int, list[int]], qubit_mem_slots: dict[int, int] | None=None, measure_name: str='measure') -> Schedule:
    """Return a schedule which measures the requested qubits according to the given
    instruction mapping and measure map, or by using the defaults provided by the backendV1.

    Args:
        qubits: List of qubits to be measured.
        backend (Union[Backend, BaseBackend]): A backend instance, which contains
            hardware-specific data required for scheduling.
        inst_map: Mapping of circuit operations to pulse schedules. If None, defaults to the
                  ``instruction_schedule_map`` of ``backend``.
        meas_map: List of sets of qubits that must be measured together. If None, defaults to
                  the ``meas_map`` of ``backend``.
        qubit_mem_slots: Mapping of measured qubit index to classical bit index.
        measure_name: Name of the measurement schedule.
    Returns:
        A measurement schedule corresponding to the inputs provided.
    Raises:
        PulseError: If both ``inst_map`` or ``meas_map``, and ``backend`` is None.
    """
    schedule = Schedule(name=f'Default measurement schedule for qubits {qubits}')
    if isinstance(meas_map, list):
        meas_map = utils.format_meas_map(meas_map)
    measure_groups = set()
    for qubit in qubits:
        measure_groups.add(tuple(meas_map[qubit]))
    for measure_group_qubits in measure_groups:
        if qubit_mem_slots is not None:
            unused_mem_slots = set(measure_group_qubits) - set(qubit_mem_slots.values())
        try:
            default_sched = inst_map.get(measure_name, measure_group_qubits)
        except exceptions.PulseError as ex:
            raise exceptions.PulseError("We could not find a default measurement schedule called '{}'. Please provide another name using the 'measure_name' keyword argument. For assistance, the instructions which are defined are: {}".format(measure_name, inst_map.instructions)) from ex
        for time, inst in default_sched.instructions:
            if inst.channel.index not in qubits:
                continue
            if qubit_mem_slots and isinstance(inst, instructions.Acquire):
                if inst.channel.index in qubit_mem_slots:
                    mem_slot = channels.MemorySlot(qubit_mem_slots[inst.channel.index])
                else:
                    mem_slot = channels.MemorySlot(unused_mem_slots.pop())
                inst = instructions.Acquire(inst.duration, inst.channel, mem_slot=mem_slot)
            schedule = schedule.insert(time, inst)
    return schedule