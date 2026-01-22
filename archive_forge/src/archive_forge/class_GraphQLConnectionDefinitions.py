from typing import Any, Dict, List, NamedTuple, Optional, Union
from graphql import (
from graphql import GraphQLNamedOutputType
class GraphQLConnectionDefinitions(NamedTuple):
    edge_type: GraphQLObjectType
    connection_type: GraphQLObjectType