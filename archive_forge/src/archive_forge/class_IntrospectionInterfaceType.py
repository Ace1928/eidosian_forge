from textwrap import dedent
from typing import Any, Dict, List, Optional, Union
from ..language import DirectiveLocation
class IntrospectionInterfaceType(WithName):
    kind: Literal['interface']
    fields: List[IntrospectionField]
    interfaces: List[SimpleIntrospectionType]
    possibleTypes: List[SimpleIntrospectionType]