from typing import Dict, List, Union, Optional, Any
from typing_extensions import Literal
from pydantic import BaseModel, Field
from rpcq.messages import TargetDevice as TargetQuantumProcessor
class Supported2QGate:
    WILDCARD = 'WILDCARD'
    CZ = 'CZ'
    ISWAP = 'ISWAP'
    CPHASE = 'CPHASE'
    XY = 'XY'