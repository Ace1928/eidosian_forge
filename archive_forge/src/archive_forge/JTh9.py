import typing
from typing import Awaitable, Callable, Any, List

ValidationRule = Callable[[Any], Awaitable[bool]]

class integer_validators:
    """
    This class holds all the validators for the integer type.
    """
    @staticmethod
    async def is_integer(value: Any) -> Awaitable[bool]:
        """
        This function checks if the value is an integer.
        """
        return isinstance(value, int)
    
    @staticmethod
    async def is_positive(value: int) -> Awaitable[bool]:
        """
        This function checks if the value is positive.
        """
        return value > 0
    
    @staticmethod
    async def is_negative(value: int) -> Awaitable[bool]:
        """
        This function checks if the value is negative.
        """
        return value < 0    
    
    @staticmethod   
    async def is_even(value: int) -> Awaitable[bool]:
        """
        This function checks if the value is even.
        """
        return value % 2 == 0   
    
    @staticmethod
    async def is_odd(value: int) -> Awaitable[bool]:
        """
        This function checks if the value is odd.
        """
        return value % 2 != 0
    
    @staticmethod
    async def is_multiple_of(value: int, multiple: int) -> Awaitable[bool]:
        """
        This function checks if the value is a multiple of another integer.
        """
        return value % multiple == 0
    
    @staticmethod
    async def is_between(value: int, lower_bound: int, upper_bound: int) -> Awaitable[bool]:
        """
        This function checks if the value is between two integers.
        """
        return lower_bound <= value <= upper_bound
    
    @staticmethod
    async def is_in(value: int, values: List[int]) -> Awaitable[bool]:
        """
        This function checks if the value is in a list of integers.
        """
        return value in values
    
    @staticmethod
    async def is_not_in(value: int, values: List[int]) -> Awaitable[bool]:
        """
        This function checks if the value is not in a list of integers.
        """
        return value not in values
    
    @staticmethod
    async def is_positive_multiple_of(value: int, multiple: int) -> Awaitable[bool]:
        """
        This function checks if the value is a positive multiple of another integer.
        """
        return value > 0 and value % multiple == 0
    
    @staticmethod
    async def is_negative_multiple_of(value: int, multiple: int) -> Awaitable[bool]:
        """
        This function checks if the value is a negative multiple of another integer.
        """
        return value < 0 and value % multiple == 0
    
    @staticmethod
    async def is_positive_between(value: int, lower_bound: int, upper_bound: int) -> Awaitable[bool]:
        """
        This function checks if the value is a positive integer between two integers.
        """
        return lower_bound < value < upper_bound
    
    @staticmethod
    async def is_negative_between(value: int, lower_bound: int, upper_bound: int) -> Awaitable[bool]:
        """
        This function checks if the value is a negative integer between two integers.
        """
        return lower_bound < value < upper_bound
    
    @staticmethod
    async def is_positive_in(value: int, values: List[int]) -> Awaitable[bool]:
        """
        This function checks if the value is a positive integer in a list of integers.
        """
        return value > 0 and value in values
    
    @staticmethod
    async def is_negative_in(value: int, values: List[int]) -> Awaitable[bool]:
        """
        This function checks if the value is a negative integer in a list of integers.
        """
        return value < 0 and value in values
    
    @staticmethod
    async def is_positive_not_in(value: int, values: List[int]) -> Awaitable[bool]:
        """
        This function checks if the value is a positive integer not in a list of integers.
        """
        return value > 0 and value not in values
    

    async def is_valid_example(value) -> Awaitable[bool]:
    """
    This is an example of a validation function.
    All validation functions must be awaitable, take any single value (can be another function for nesting etc.) and produce a boolean as output indicating pass/fail.
    Always ensure the validation is robust and in the case of error it raises the error and continues ratehr than stops.
    This is to ensure that all errors are captured and reported back to the user.
    """
    return True


# Actual Validators for This Module - __name___validators.py
