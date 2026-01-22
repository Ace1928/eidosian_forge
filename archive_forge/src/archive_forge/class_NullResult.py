from collections.abc import Mapping
from inspect import iscoroutinefunction
from typing import Any, Callable, Dict, Optional
from graphql import (
from graphql.pyutils import AwaitableOrValue
class NullResult:

    def __init__(self, clientMutationId: Optional[str]=None) -> None:
        self.clientMutationId = clientMutationId