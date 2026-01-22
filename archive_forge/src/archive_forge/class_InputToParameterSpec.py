from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
@dataclass
class InputToParameterSpec:
    arg: TensorArgument
    parameter_name: str