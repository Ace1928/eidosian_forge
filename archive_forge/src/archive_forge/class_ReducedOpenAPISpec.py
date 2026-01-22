from dataclasses import dataclass
from typing import List, Tuple
from langchain_core.utils.json_schema import dereference_refs
@dataclass(frozen=True)
class ReducedOpenAPISpec:
    """A reduced OpenAPI spec.

    This is a quick and dirty representation for OpenAPI specs.

    Attributes:
        servers: The servers in the spec.
        description: The description of the spec.
        endpoints: The endpoints in the spec.
    """
    servers: List[dict]
    description: str
    endpoints: List[Tuple[str, str, dict]]