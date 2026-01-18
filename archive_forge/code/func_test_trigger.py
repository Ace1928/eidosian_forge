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
def test_trigger():
    accelerator = Accelerator()
    assert accelerator.check_trigger() is False
    if accelerator.is_main_process:
        accelerator.set_trigger()
    assert accelerator.check_trigger() is True
    assert accelerator.check_trigger() is False