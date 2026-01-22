import locale
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from .namespaces import XQT_ERRORS_NAMESPACE
from .datatypes import QName
class ElementPathRuntimeError(ElementPathError, RuntimeError):
    pass