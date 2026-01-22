from typing import get_origin, get_args, List, Dict


def do_validate_generic(self, value, expected_type):
    """Template function for validating generic types."""
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if origin is List:
        if not all(isinstance(item, args[0]) for item in value):
            raise ValueError("List items do not match expected type")
    elif origin is Dict:
        key_type, value_type = args
        if not all(
            isinstance(k, key_type) and isinstance(v, value_type)
            for k, v in value.items()
        ):

def do_validate_list(self, value):
    """Validates if the value is a list."""
    if not isinstance(value, list):
        raise ValueError(f"Expected list, got {type(value).__name__}")


def do_validate_dict(self, value):
    """Validates if the value is a dictionary."""
    if not isinstance(value, dict):
        raise ValueError(f"Expected dict, got {type(value).__name__}")


def do_validate_int(self, value):
    """Validates if the value is an integer."""
    if not isinstance(value, int):
        raise ValueError(f"Expected int, got {type(value).__name__}")


def do_validate_str(self, value):
    """Validates if the value is a string."""
    if not isinstance(value, str):
        raise ValueError(f"Expected str, got {type(value).__name__}")
