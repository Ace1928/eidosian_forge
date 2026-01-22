from textwrap import dedent
from typing import Any, Dict, List, Optional, Union
from ..language import DirectiveLocation
class IntrospectionQuery(TypedDict):
    """The root typed dictionary for schema introspections."""
    __schema: IntrospectionSchema