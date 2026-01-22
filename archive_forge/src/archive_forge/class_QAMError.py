from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Mapping, Optional, TypeVar
import numpy as np
from pyquil.api._abstract_compiler import QuantumExecutable
class QAMError(RuntimeError):
    pass