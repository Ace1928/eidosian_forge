from textwrap import dedent
from typing import Any, Dict, List, Optional, Union
from ..language import DirectiveLocation
class MaybeWithDeprecated(TypedDict, total=False):
    isDeprecated: bool
    deprecationReason: Optional[str]