from typing import Any, Type, Optional, Set, Dict
class WrongTypeError(DaciteFieldError):

    def __init__(self, field_type: Type, value: Any, field_path: Optional[str]=None) -> None:
        super().__init__(field_path=field_path)
        self.field_type = field_type
        self.value = value

    def __str__(self) -> str:
        return f'wrong value type for field "{self.field_path}" - should be "{_name(self.field_type)}" instead of value "{self.value}" of type "{_name(type(self.value))}"'