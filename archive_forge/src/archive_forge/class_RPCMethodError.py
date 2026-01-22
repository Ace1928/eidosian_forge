import uuid
import warnings
from inspect import signature
from typing import Callable, Optional, Tuple, Union, List, Any
import rpcq.messages
class RPCMethodError(AttributeError):
    """JSON RPC error that is raised by JSON RPC spec for nonexistent methods"""