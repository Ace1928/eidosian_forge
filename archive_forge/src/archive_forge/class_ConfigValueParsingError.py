from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import (
class ConfigValueParsingError(ConfigError):
    """Raised when a configuration value cannot be parsed."""

    def __init__(self, name, value):
        super().__init__(f'Config option {name}: value cannot be parsed (given {repr(value)})')