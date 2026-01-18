from __future__ import annotations as _annotations
import sys
from dataclasses import is_dataclass
from typing import TYPE_CHECKING, Any, Dict, Generic, Iterable, Set, TypeVar, Union, cast, final, overload
from pydantic_core import CoreSchema, SchemaSerializer, SchemaValidator, Some
from typing_extensions import Literal, get_args, is_typeddict
from pydantic.errors import PydanticUserError
from pydantic.main import BaseModel
from ._internal import _config, _generate_schema, _typing_extra
from .config import ConfigDict
from .json_schema import (
from .plugin._schema_validator import create_schema_validator
Generate a JSON schema including definitions from multiple type adapters.

        Args:
            __inputs: Inputs to schema generation. The first two items will form the keys of the (first)
                output mapping; the type adapters will provide the core schemas that get converted into
                definitions in the output JSON schema.
            by_alias: Whether to use alias names.
            title: The title for the schema.
            description: The description for the schema.
            ref_template: The format string used for generating $ref strings.
            schema_generator: The generator class used for creating the schema.

        Returns:
            A tuple where:

                - The first element is a dictionary whose keys are tuples of JSON schema key type and JSON mode, and
                    whose values are the JSON schema corresponding to that pair of inputs. (These schemas may have
                    JsonRef references to definitions that are defined in the second returned element.)
                - The second element is a JSON schema containing all definitions referenced in the first returned
                    element, along with the optional title and description keys.

        