import argparse
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
def to_unsigned(type_str) -> str:
    if type_str == 'int32':
        return 'uint32'
    elif type_str == 'int64':
        return 'uint64'
    else:
        return type_str