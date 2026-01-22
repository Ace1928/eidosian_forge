from typing import Mapping, TypeVar
from .error_reporting import ValidationError
class RedefiningStaticFieldAsDynamic(ValidationError):
    """According to PEP 621:

    Build back-ends MUST raise an error if the metadata specifies a field
    statically as well as being listed in dynamic.
    """