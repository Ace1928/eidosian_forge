from __future__ import annotations
import logging
import sys
from typing import Final
def update_formatter() -> None:
    for log in _loggers.values():
        setup_formatter(log)