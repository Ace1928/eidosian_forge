from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterator
from qiskit import pulse, circuit
from qiskit.visualization.pulse_v2.types import PhaseFreqTuple, PulseInstruction
@classmethod
def _calculate_current_frame(cls, frame_changes: list[pulse.instructions.Instruction], phase: float, frequency: float) -> tuple[float, float]:
    """Calculate the current frame from the previous frame.

        If parameter is unbound phase or frequency accumulation with this instruction is skipped.

        Args:
            frame_changes: List of frame change instructions at a specific time.
            phase: Phase of previous frame.
            frequency: Frequency of previous frame.

        Returns:
            Phase and frequency of new frame.
        """
    for frame_change in frame_changes:
        if isinstance(frame_change, pulse.instructions.SetFrequency):
            frequency = frame_change.frequency
        elif isinstance(frame_change, pulse.instructions.ShiftFrequency):
            frequency += frame_change.frequency
        elif isinstance(frame_change, pulse.instructions.SetPhase):
            phase = frame_change.phase
        elif isinstance(frame_change, pulse.instructions.ShiftPhase):
            phase += frame_change.phase
    return (phase, frequency)