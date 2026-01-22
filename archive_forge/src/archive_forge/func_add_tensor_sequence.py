import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_tensor_sequence(self, jitval, values):
    assert jitval not in self.tensor_sequences
    self.tensor_sequences[jitval] = values