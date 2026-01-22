from typing import List, Union
class SortDirection:
    """
    This special class is used to indicate sort direction.
    """
    DIRSTRING = None

    def __init__(self, field: str) -> None:
        self.field = field