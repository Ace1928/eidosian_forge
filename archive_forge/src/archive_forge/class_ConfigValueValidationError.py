from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import (
class ConfigValueValidationError(ConfigError):
    """Raised when a configuration value cannot be validated."""

    def __init__(self, name, value):
        super().__init__(f'Config option {name}: value is invalid (given {repr(value)})')