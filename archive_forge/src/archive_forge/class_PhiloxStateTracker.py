import functools
from collections import defaultdict
from typing import Callable, Dict
import torch
import torch._decomp as decomp
from torch._decomp import get_decompositions
from torch._ops import OpOverload
class PhiloxStateTracker:
    """
    Singleton class to track the philox rng state during AOT Autograd tracing.
    For each aot tracing instance, AOT Autograd resets this tracker and keeps
    track of both forward and backward offsets. At runtime, we only care about
    the total consumed forward and backward offsets. For dynamic shapes, these
    offsets are a function of input shapes. Therefore, the AOT generated graphs
    have additional outputs that compute total consumed forward and backward
    offsets.
    """
    running_state: PhiloxState
    fwd_state: PhiloxState
    bwd_state: PhiloxState

    def __enter__(self):
        PhiloxStateTracker.reset()
        return self

    def __exit__(self, exc_type, exc_cal, exc_tb):
        PhiloxStateTracker.reset()

    @classmethod
    def reset(cls):
        cls.running_state = PhiloxState()
        cls.fwd_state = PhiloxState()
        cls.bwd_state = PhiloxState()

    @classmethod
    def mark_beginning_of_forward(cls):
        cls.running_state = cls.fwd_state

    @classmethod
    def mark_beginning_of_backward(cls):
        cls.running_state = cls.bwd_state

    @classmethod
    def record_state(cls, seed, offset, mode):
        if mode == 'forward':
            cls.fwd_state.set_state(seed, offset)
            cls.mark_beginning_of_forward()
        else:
            assert mode == 'backward'
            cls.bwd_state.set_state(seed, offset)

    @classmethod
    def get_state_as_tensor(cls):
        return cls.running_state.get_state_as_tensor()

    @classmethod
    def get_state_as_tuple(cls):
        return cls.running_state.get_state_as_tuple()

    @classmethod
    def set_state_from_tensor(cls, x):
        cls.running_state.set_state_from_tensor(x)

    @classmethod
    def advance_offset(cls, consumed_offset):
        cls.running_state.advance_offset(consumed_offset)

    @classmethod
    def get_current_relative_offset(cls):
        return cls.running_state.relative_offset

    @staticmethod
    def multiple_of_4(offset):
        return (offset + 3) // 4 * 4

    @classmethod
    def get_updated_fwd_offset(cls):
        if not cls.fwd_state.offset_advanced_alteast_once:
            return cls.fwd_state.base_offset
        return cls.multiple_of_4(cls.fwd_state.base_offset + cls.fwd_state.relative_offset)

    @classmethod
    def get_updated_bwd_offset(cls):
        if not cls.bwd_state.offset_advanced_alteast_once:
            return cls.bwd_state.base_offset
        return cls.multiple_of_4(cls.bwd_state.base_offset + cls.bwd_state.relative_offset)