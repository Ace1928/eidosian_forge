import abc
import dataclasses
import functools
import inspect
import sys
from dataclasses import Field, fields
from typing import Any, Callable, Dict, Optional, Tuple, Union, Type, get_type_hints
from enum import Enum
from marshmallow.exceptions import ValidationError  # type: ignore
from dataclasses_json.utils import CatchAllVar
class _UndefinedParameterAction(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def handle_from_dict(cls, kvs: Dict[Any, Any]) -> Dict[str, Any]:
        """
        Return the parameters to initialize the class with.
        """
        pass

    @staticmethod
    def handle_to_dict(obj, kvs: Dict[Any, Any]) -> Dict[Any, Any]:
        """
        Return the parameters that will be written to the output dict
        """
        return kvs

    @staticmethod
    def handle_dump(obj) -> Dict[Any, Any]:
        """
        Return the parameters that will be added to the schema dump.
        """
        return {}

    @staticmethod
    def create_init(obj) -> Callable:
        return obj.__init__

    @staticmethod
    def _separate_defined_undefined_kvs(cls, kvs: Dict) -> Tuple[KnownParameters, UnknownParameters]:
        """
        Returns a 2 dictionaries: defined and undefined parameters
        """
        class_fields = fields(cls)
        field_names = [field.name for field in class_fields]
        unknown_given_parameters = {k: v for k, v in kvs.items() if k not in field_names}
        known_given_parameters = {k: v for k, v in kvs.items() if k in field_names}
        return (known_given_parameters, unknown_given_parameters)