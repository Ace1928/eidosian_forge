from typing import Any, Type, Optional, Set, Dict
def update_path(self, parent_field_path: str) -> None:
    if self.field_path:
        self.field_path = f'{parent_field_path}.{self.field_path}'
    else:
        self.field_path = parent_field_path