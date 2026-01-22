from typing import Any, Callable, NamedTuple, Optional, Union
from graphql_relay.utils.base64 import base64, unbase64
from graphql import (
class ResolvedGlobalId(NamedTuple):
    type: str
    id: str