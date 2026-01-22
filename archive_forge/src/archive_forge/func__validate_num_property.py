from typing import TYPE_CHECKING
from typing import List
from typing import Optional
def _validate_num_property(self, property_name: str, value: float) -> None:
    """Helper function to validate some of the properties."""
    if not isinstance(value, (int, float)):
        raise ValueError(f'{property_name} should be an integer or a float')
    if value < 0:
        raise ValueError(f'{property_name} cannot be less then 0')