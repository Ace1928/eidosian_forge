from __future__ import annotations
import typing
from cryptography import utils
from cryptography.exceptions import (
from cryptography.hazmat.primitives import (
from cryptography.hazmat.primitives.kdf import KeyDerivationFunction
class CounterLocation(utils.Enum):
    BeforeFixed = 'before_fixed'
    AfterFixed = 'after_fixed'
    MiddleFixed = 'middle_fixed'