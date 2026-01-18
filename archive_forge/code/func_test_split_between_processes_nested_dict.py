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
def test_split_between_processes_nested_dict():
    state = AcceleratorState()
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    c = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    if state.num_processes in (1, 2, 4):
        data = {'a': a, 'b': b, 'c': c}
        data_copy = deepcopy(data)
        with state.split_between_processes(data) as results:
            if state.process_index == 0:
                assert results['a'] == data_copy['a'][:8 // state.num_processes]
            elif state.num_processes == 2:
                assert results['a'] == data_copy['a'][4:]
            elif state.process_index == 3:
                assert results['a'] == data_copy['a'][-2:], f'Expected: {data_copy['a'][-2]}, Actual: {results['a']}'
            if state.process_index == 0:
                assert results['b'] == data_copy['b'][:8 // state.num_processes]
            elif state.num_processes == 2:
                assert results['b'] == data_copy['b'][4:]
            elif state.process_index == 3:
                assert results['b'] == data_copy['b'][-2:]
            if state.process_index == 0:
                assert torch.allclose(results['c'], data_copy['c'][:8 // state.num_processes]), f'Did not obtain expected values on process 0, expected `{data['c'][:8 // state.num_processes]}`, received: {results['c']}'
            elif state.num_processes == 2:
                assert torch.allclose(results['c'], data_copy['c'][4:]), f'Did not obtain expected values on process 2, expected `{data['c'][4:]}`, received: {results['c']}'
            elif state.process_index == 3:
                assert torch.allclose(results['c'], data_copy['c'][-2:]), f'Did not obtain expected values on process 4, expected `{data['c'][-2:]}`, received: {results['c']}'
    state.wait_for_everyone()