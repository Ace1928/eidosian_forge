from textwrap import dedent
from typing import Any, Dict, List, Optional, Union
from ..language import DirectiveLocation
class WithDeprecated(TypedDict):
    isDeprecated: bool
    deprecationReason: Optional[str]