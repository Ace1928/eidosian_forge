from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING
from qiskit.pulse import channels, exceptions, instructions, utils
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.schedule import Schedule
from qiskit.providers.backend import BackendV2
def _measure_v2(qubits: Sequence[int], target: Target, meas_map: list[list[int]] | dict[int, list[int]], qubit_mem_slots: dict[int, int], measure_name: str='measure') -> Schedule:
    """Return a schedule which measures the requested qubits according to the given
    target and measure map, or by using the defaults provided by the backendV2.

    Args:
        qubits: List of qubits to be measured.
        target: The :class:`~.Target` representing the target backend.
        meas_map: List of sets of qubits that must be measured together.
        qubit_mem_slots: Mapping of measured qubit index to classical bit index.
        measure_name: Name of the measurement schedule.

    Returns:
        A measurement schedule corresponding to the inputs provided.
    """
    schedule = Schedule(name=f'Default measurement schedule for qubits {qubits}')
    if isinstance(meas_map, list):
        meas_map = utils.format_meas_map(meas_map)
    meas_group = set()
    for qubit in qubits:
        meas_group |= set(meas_map[qubit])
    meas_group = sorted(meas_group)
    meas_group_set = set(range(max(meas_group) + 1))
    unassigned_qubit_indices = sorted(set(meas_group) - qubit_mem_slots.keys())
    unassigned_reg_indices = sorted(meas_group_set - set(qubit_mem_slots.values()), reverse=True)
    if set(qubit_mem_slots.values()).issubset(meas_group_set):
        for qubit in unassigned_qubit_indices:
            qubit_mem_slots[qubit] = unassigned_reg_indices.pop()
    for measure_qubit in meas_group:
        try:
            if measure_qubit in qubits:
                default_sched = target.get_calibration(measure_name, (measure_qubit,)).filter(channels=[channels.MeasureChannel(measure_qubit), channels.AcquireChannel(measure_qubit)])
                schedule += _schedule_remapping_memory_slot(default_sched, qubit_mem_slots)
        except KeyError as ex:
            raise exceptions.PulseError("We could not find a default measurement schedule called '{}'. Please provide another name using the 'measure_name' keyword argument. For assistance, the instructions which are defined are: {}".format(measure_name, target.instructions)) from ex
    return schedule