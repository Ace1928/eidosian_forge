from __future__ import annotations
import typing
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Type
import numpy as np
from qiskit.pulse import channels as chans, exceptions, instructions
from qiskit.pulse.channels import ClassicalIOChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.exceptions import UnassignedDurationError
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.instructions import directives
from qiskit.pulse.schedule import Schedule, ScheduleBlock, ScheduleComponent
def add_implicit_acquires(schedule: ScheduleComponent, meas_map: list[list[int]]) -> Schedule:
    """Return a new schedule with implicit acquires from the measurement mapping replaced by
    explicit ones.

    .. warning:: Since new acquires are being added, Memory Slots will be set to match the
                 qubit index. This may overwrite your specification.

    Args:
        schedule: Schedule to be aligned.
        meas_map: List of lists of qubits that are measured together.

    Returns:
        A ``Schedule`` with the additional acquisition instructions.
    """
    new_schedule = Schedule.initialize_from(schedule)
    acquire_map = {}
    for time, inst in schedule.instructions:
        if isinstance(inst, instructions.Acquire):
            if inst.mem_slot and inst.mem_slot.index != inst.channel.index:
                warnings.warn("One of your acquires was mapped to a memory slot which didn't match the qubit index. I'm relabeling them to match.")
            all_qubits = []
            for sublist in meas_map:
                if inst.channel.index in sublist:
                    all_qubits.extend(sublist)
            for i in all_qubits:
                explicit_inst = instructions.Acquire(inst.duration, chans.AcquireChannel(i), mem_slot=chans.MemorySlot(i), kernel=inst.kernel, discriminator=inst.discriminator)
                if time not in acquire_map:
                    new_schedule.insert(time, explicit_inst, inplace=True)
                    acquire_map = {time: {i}}
                elif i not in acquire_map[time]:
                    new_schedule.insert(time, explicit_inst, inplace=True)
                    acquire_map[time].add(i)
        else:
            new_schedule.insert(time, inst, inplace=True)
    return new_schedule