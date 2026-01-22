from typing import Any, Dict, List, NamedTuple, Optional, Union
from graphql import (
from graphql import GraphQLNamedOutputType
class ConnectionConstructor(Protocol):

    def __call__(self, *, edges: List[EdgeType], pageInfo: PageInfoType) -> ConnectionType:
        ...