import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union
from collections import OrderedDict
import yaml
def _handle_local_pip_requirement_file(pip_file: str):
    pip_path = Path(pip_file)
    if not pip_path.is_file():
        raise ValueError(f'{pip_path} is not a valid file')
    return pip_path.read_text().strip().split('\n')