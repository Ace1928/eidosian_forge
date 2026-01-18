import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from packaging import version
from ..utils import is_auto_awq_available, is_torch_available, logging
def to_dict_optimum(self):
    """
        Get compatible dict for optimum gptq config
        """
    quant_dict = self.to_dict()
    quant_dict['disable_exllama'] = not self.use_exllama
    return quant_dict