import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
class ObsoleteFactoryWrapper:

    def __init__(self, name):
        self.name = name

    def calc(self, **kwargs):
        from ase.calculators.calculator import get_calculator_class
        cls = get_calculator_class(self.name)
        return cls(**kwargs)