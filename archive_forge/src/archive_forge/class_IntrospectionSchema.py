from textwrap import dedent
from typing import Any, Dict, List, Optional, Union
from ..language import DirectiveLocation
class IntrospectionSchema(MaybeWithDescription):
    queryType: IntrospectionObjectType
    mutationType: Optional[IntrospectionObjectType]
    subscriptionType: Optional[IntrospectionObjectType]
    types: List[IntrospectionType]
    directives: List[IntrospectionDirective]