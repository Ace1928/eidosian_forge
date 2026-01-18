import itertools
import types
import warnings
from collections import defaultdict
from typing import (
import numpy as np
from rpcq.messages import NativeQuilMetadata, ParameterAref
from pyquil._parser.parser import run_parser
from pyquil._memory import Memory
from pyquil.gates import MEASURE, RESET, MOVE
from pyquil.noise import _check_kraus_ops, _create_kraus_pragmas, pauli_kraus_map
from pyquil.quilatom import (
from pyquil.quilbase import (
from pyquil.quiltcalibrations import (
def validate_protoquil(program: Program, quilt: bool=False) -> None:
    """
    Ensure that a program is valid ProtoQuil or Quil-T, otherwise raise a ValueError.
    Protoquil is a subset of Quil which excludes control flow and classical instructions.

    :param quilt: Validate the program as Quil-T.
    :param program: The Quil program to validate.
    """
    '\n    Ensure that a program is valid ProtoQuil, otherwise raise a ValueError.\n    Protoquil is a subset of Quil which excludes control flow and classical instructions.\n\n    :param program: The Quil program to validate.\n    '
    if quilt:
        valid_instruction_types = tuple([Pragma, Declare, Halt, Gate, Measurement, Reset, ResetQubit, DelayQubits, DelayFrames, Fence, FenceAll, ShiftFrequency, SetFrequency, SetScale, ShiftPhase, SetPhase, SwapPhases, Pulse, Capture, RawCapture, DefCalibration, DefFrame, DefMeasureCalibration, DefWaveform])
    else:
        valid_instruction_types = tuple([Pragma, Declare, Gate, Reset, ResetQubit, Measurement])
        if program.calibrations:
            raise ValueError('ProtoQuil validation failed: Quil-T calibrations are not allowed.')
        if program.waveforms:
            raise ValueError('ProtoQuil validation failed: Quil-T waveform definitions are not allowed.')
        if program.frames:
            raise ValueError('ProtoQuil validation failed: Quil-T frame definitions are not allowed.')
    for instr in program.instructions:
        if not isinstance(instr, valid_instruction_types):
            raise ValueError(f'ProtoQuil validation failed: {instr} is not allowed.')