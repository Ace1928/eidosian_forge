from typing import TYPE_CHECKING
from typing import List
from typing import Optional
class _MarginSettingsDescriptor:
    """Descriptor which validates below attributes.

    - top
    - bottom
    - left
    - right
    """

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, cls) -> Optional[float]:
        return obj._margin.get(self.name, None)

    def __set__(self, obj, value) -> None:
        getattr(obj, '_validate_num_property')(f'Margin {self.name}', value)
        obj._margin[self.name] = value
        obj._print_options['margin'] = obj._margin