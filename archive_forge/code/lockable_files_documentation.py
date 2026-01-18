from typing import Optional, Type, Union
from . import counted_lock, errors, lock, transactions, urlutils
from .decorators import only_raises
from .transport import Transport
Create lock mechanism