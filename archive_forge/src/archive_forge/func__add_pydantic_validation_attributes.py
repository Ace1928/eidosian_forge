import copy
import dataclasses
import sys
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Generator, Optional, Type, TypeVar, Union, overload
from typing_extensions import dataclass_transform
from .class_validators import gather_all_validators
from .config import BaseConfig, ConfigDict, Extra, get_config
from .error_wrappers import ValidationError
from .errors import DataclassTypeError
from .fields import Field, FieldInfo, Required, Undefined
from .main import create_model, validate_model
from .utils import ClassAttribute
def _add_pydantic_validation_attributes(dc_cls: Type['Dataclass'], config: Type[BaseConfig], validate_on_init: bool, dc_cls_doc: str) -> None:
    """
    We need to replace the right method. If no `__post_init__` has been set in the stdlib dataclass
    it won't even exist (code is generated on the fly by `dataclasses`)
    By default, we run validation after `__init__` or `__post_init__` if defined
    """
    init = dc_cls.__init__

    @wraps(init)
    def handle_extra_init(self: 'Dataclass', *args: Any, **kwargs: Any) -> None:
        if config.extra == Extra.ignore:
            init(self, *args, **{k: v for k, v in kwargs.items() if k in self.__dataclass_fields__})
        elif config.extra == Extra.allow:
            for k, v in kwargs.items():
                self.__dict__.setdefault(k, v)
            init(self, *args, **{k: v for k, v in kwargs.items() if k in self.__dataclass_fields__})
        else:
            init(self, *args, **kwargs)
    if hasattr(dc_cls, '__post_init__'):
        try:
            post_init = dc_cls.__post_init__.__wrapped__
        except AttributeError:
            post_init = dc_cls.__post_init__

        @wraps(post_init)
        def new_post_init(self: 'Dataclass', *args: Any, **kwargs: Any) -> None:
            if config.post_init_call == 'before_validation':
                post_init(self, *args, **kwargs)
            if self.__class__.__pydantic_run_validation__:
                self.__pydantic_validate_values__()
                if hasattr(self, '__post_init_post_parse__'):
                    self.__post_init_post_parse__(*args, **kwargs)
            if config.post_init_call == 'after_validation':
                post_init(self, *args, **kwargs)
        setattr(dc_cls, '__init__', handle_extra_init)
        setattr(dc_cls, '__post_init__', new_post_init)
    else:

        @wraps(init)
        def new_init(self: 'Dataclass', *args: Any, **kwargs: Any) -> None:
            handle_extra_init(self, *args, **kwargs)
            if self.__class__.__pydantic_run_validation__:
                self.__pydantic_validate_values__()
            if hasattr(self, '__post_init_post_parse__'):
                initvars_and_values: Dict[str, Any] = {}
                for i, f in enumerate(self.__class__.__dataclass_fields__.values()):
                    if f._field_type is dataclasses._FIELD_INITVAR:
                        try:
                            initvars_and_values[f.name] = args[i]
                        except IndexError:
                            initvars_and_values[f.name] = kwargs.get(f.name, f.default)
                self.__post_init_post_parse__(**initvars_and_values)
        setattr(dc_cls, '__init__', new_init)
    setattr(dc_cls, '__pydantic_run_validation__', ClassAttribute('__pydantic_run_validation__', validate_on_init))
    setattr(dc_cls, '__pydantic_initialised__', False)
    setattr(dc_cls, '__pydantic_model__', create_pydantic_model_from_dataclass(dc_cls, config, dc_cls_doc))
    setattr(dc_cls, '__pydantic_validate_values__', _dataclass_validate_values)
    setattr(dc_cls, '__validate__', classmethod(_validate_dataclass))
    setattr(dc_cls, '__get_validators__', classmethod(_get_validators))
    if dc_cls.__pydantic_model__.__config__.validate_assignment and (not dc_cls.__dataclass_params__.frozen):
        setattr(dc_cls, '__setattr__', _dataclass_validate_assignment_setattr)