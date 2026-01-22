import sys
import types
import typing
from typing import (
from weakref import WeakKeyDictionary, WeakValueDictionary
from typing_extensions import Annotated, Literal as ExtLiteral
from .class_validators import gather_all_validators
from .fields import DeferredType
from .main import BaseModel, create_model
from .types import JsonWrapper
from .typing import display_as_type, get_all_type_hints, get_args, get_origin, typing_base
from .utils import all_identical, lenient_issubclass
@classmethod
def __concrete_name__(cls: Type[Any], params: Tuple[Type[Any], ...]) -> str:
    """Compute class name for child classes.

        :param params: Tuple of types the class . Given a generic class
            `Model` with 2 type variables and a concrete model `Model[str, int]`,
            the value `(str, int)` would be passed to `params`.
        :return: String representing a the new class where `params` are
            passed to `cls` as type variables.

        This method can be overridden to achieve a custom naming scheme for GenericModels.
        """
    param_names = [display_as_type(param) for param in params]
    params_component = ', '.join(param_names)
    return f'{cls.__name__}[{params_component}]'