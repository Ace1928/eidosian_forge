from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
@dataclass
class GradientToUserInputSpec:
    arg: TensorArgument
    user_input_name: str