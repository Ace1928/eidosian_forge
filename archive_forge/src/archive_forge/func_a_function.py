from unittest.mock import patch
from docstring_parser import parse_from_object
def a_function(param1: str, param2: int=2):
    """Short description
        Args:
            param1: Description for param1
            param2: Description for param2
        """
    return f'{param1} {param2}'