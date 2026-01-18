from typing import Any, Callable, List, Optional, Union, cast
from ..language import (
from ..pyutils import Undefined
from ..type import (
from .type_from_ast import type_from_ast
Initialize the TypeInfo for the given GraphQL schema.

        Initial type may be provided in rare cases to facilitate traversals beginning
        somewhere other than documents.

        The optional last parameter is deprecated and will be removed in v3.3.
        