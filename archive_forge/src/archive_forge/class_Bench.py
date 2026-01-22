import argparse
import contextlib
import dataclasses
import enum
import multiprocessing
import os
import random
from collections import deque
from statistics import mean, stdev
from typing import Callable
import torch
@dataclasses.dataclass
class Bench:
    ag: Callable[[], None]
    rs: Callable[[], None]

    def __getitem__(self, step: Step):
        if step is Step.AllGather:
            return self.ag
        elif step is Step.ReduceScatter:
            return self.rs
        else:
            raise KeyError(f'{step}')