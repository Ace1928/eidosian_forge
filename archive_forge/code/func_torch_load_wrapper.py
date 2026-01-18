import copy
import io
from typing import List, Union
import torch
def torch_load_wrapper(f, *args, **kwargs):
    self.paths.append(f)
    return self.torch_load(f, *args, **kwargs)