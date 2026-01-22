import inspect
import re
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
def _make_export_case(m, name, configs):
    if inspect.isclass(m):
        if not issubclass(m, torch.nn.Module):
            raise TypeError('Export case class should be a torch.nn.Module.')
        m = m()
    if 'description' not in configs:
        assert m.__doc__ is not None, f'Could not find description or docstring for export case: {m}'
        configs = {**configs, 'description': m.__doc__}
    return ExportCase(**{**configs, 'model': m, 'name': name})