from textwrap import dedent
from typing import Any, Dict, List, Optional, Union
from ..language import DirectiveLocation
class IntrospectionDirective(WithName, MaybeWithIsRepeatable):
    locations: List[DirectiveLocation]
    args: List[IntrospectionInputValue]