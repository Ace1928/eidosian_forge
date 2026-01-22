from __future__ import annotations
import dataclasses
from functools import partialmethod
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
from pydantic_core import PydanticUndefined, core_schema
from pydantic_core import core_schema as _core_schema
from typing_extensions import Annotated, Literal, TypeAlias
from . import PydanticUndefinedAnnotation
from ._internal import _decorators, _internal_dataclass
from .annotated_handlers import GetCoreSchemaHandler
@dataclasses.dataclass(**_internal_dataclass.slots_true, frozen=True)
class PlainSerializer:
    """Plain serializers use a function to modify the output of serialization.

    This is particularly helpful when you want to customize the serialization for annotated types.
    Consider an input of `list`, which will be serialized into a space-delimited string.

    ```python
    from typing import List

    from typing_extensions import Annotated

    from pydantic import BaseModel, PlainSerializer

    CustomStr = Annotated[
        List, PlainSerializer(lambda x: ' '.join(x), return_type=str)
    ]

    class StudentModel(BaseModel):
        courses: CustomStr

    student = StudentModel(courses=['Math', 'Chemistry', 'English'])
    print(student.model_dump())
    #> {'courses': 'Math Chemistry English'}
    ```

    Attributes:
        func: The serializer function.
        return_type: The return type for the function. If omitted it will be inferred from the type annotation.
        when_used: Determines when this serializer should be used. Accepts a string with values `'always'`,
            `'unless-none'`, `'json'`, and `'json-unless-none'`. Defaults to 'always'.
    """
    func: core_schema.SerializerFunction
    return_type: Any = PydanticUndefined
    when_used: Literal['always', 'unless-none', 'json', 'json-unless-none'] = 'always'

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """Gets the Pydantic core schema.

        Args:
            source_type: The source type.
            handler: The `GetCoreSchemaHandler` instance.

        Returns:
            The Pydantic core schema.
        """
        schema = handler(source_type)
        try:
            return_type = _decorators.get_function_return_type(self.func, self.return_type, handler._get_types_namespace())
        except NameError as e:
            raise PydanticUndefinedAnnotation.from_name_error(e) from e
        return_schema = None if return_type is PydanticUndefined else handler.generate_schema(return_type)
        schema['serialization'] = core_schema.plain_serializer_function_ser_schema(function=self.func, info_arg=_decorators.inspect_annotated_serializer(self.func, 'plain'), return_schema=return_schema, when_used=self.when_used)
        return schema