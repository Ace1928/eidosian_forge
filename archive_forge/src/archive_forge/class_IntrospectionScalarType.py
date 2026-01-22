from textwrap import dedent
from typing import Any, Dict, List, Optional, Union
from ..language import DirectiveLocation
class IntrospectionScalarType(WithName, MaybeWithSpecifiedByUrl):
    kind: Literal['scalar']