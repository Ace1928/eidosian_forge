import contextlib
import io
import math
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.data_loader import SeedableRandomSampler, prepare_data_loader
from accelerate.state import AcceleratorState
from accelerate.test_utils import RegressionDataset, are_the_same_tensors
from accelerate.utils import (
def test_split_between_processes_list():
    state = AcceleratorState()
    data = list(range(0, 2 * state.num_processes))
    with state.split_between_processes(data) as results:
        assert len(results) == 2, f'Each process did not have two items. Process index: {state.process_index}; Length: {len(results)}'
    data = list(range(0, 3 * state.num_processes - 1))
    with state.split_between_processes(data, apply_padding=True) as results:
        if state.is_last_process:
            num_samples_per_device = math.ceil(len(data) / state.num_processes)
            assert len(results) == num_samples_per_device, f'Last process did not get the extra item(s). Process index: {state.process_index}; Length: {len(results)}'
    state.wait_for_everyone()