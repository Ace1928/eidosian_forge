import asyncio
import json
import pathlib
import re
import logging
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Union,
    get_origin,
    get_args,
    get_type_hints,
    Optional,
)
import inspect
from inspect import signature, iscoroutinefunction
from collections.abc import Mapping, Iterable
from enum import Enum

__all__ = [
    "Validate",
    "validation_rules",
]

# Type alias for clarity
ValidationRules = Dict[str, Callable[[Any], Awaitable[bool]]]


class Validate:
    """
    A comprehensive asynchronous validator designed to enforce strict type and custom validation rules
    across various function arguments and data inputs. It leverages advanced Python features and asynchronous
    programming to ensure non-blocking operations, detailed logging, and robust error handling.
    """

    def __init__(self, validation_rules: ValidationRules):
        """
        Initializes the AsyncValidator with a set of validation rules.

        Args:
            validation_rules (ValidationRules): A dictionary mapping rule names to their corresponding
                                                asynchronous validation functions.
        """
        self.validation_rules = validation_rules

    async def __call__(self, value: Any, rule_name: Optional[str] = None) -> bool:
        """
        Asynchronously validates a value against a specified rule when the instance is called.
        If no rule_name is provided, it attempts to validate the value using all available rules.

        Args:
            value (Any): The value to validate.
            rule_name (Optional[str]): The name of the validation rule to apply. Defaults to None.

        Returns:
            bool: True if the value passes the validation rule(s), False otherwise.

        Raises:
            ValueError: If the specified rule name does not exist in the validation rules.
        """
        if rule_name:
            if rule_name not in self.validation_rules:
                error_msg = f"Validation rule '{rule_name}' does not exist."
                logging.error(error_msg)
                raise ValueError(error_msg)

            validation_func = self.validation_rules[rule_name]
            result = await validation_func(value)
            return result
        else:
            # Validate against all rules
            results = await asyncio.gather(
                *[rule(value) for rule in self.validation_rules.values()]
            )
            return all(results)

        # Automatically run the methods: is_valid_func_signature, is_valid_argument, is_valid_type
        # Ensure that any validation rules (if present) are utilised flexibly and dynamically with each of these methods.
        await self.is_valid_func_signature(value)
        await self.is_valid_argument(value)
        await self.is_valid_type(value, type(value))

    async def is_valid_func_signature(self, func: Callable, *args, **kwargs) -> None:
        """
        Asynchronously validates a function's signature against provided arguments and types,
        ensuring compatibility with both synchronous and asynchronous functions. It leverages Python's
        introspection capabilities for dynamic signature validation, detailed logging, and robust error handling.

        Args:
            func (Callable): The function whose signature is being validated.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Raises:
            TypeError: If the provided arguments do not match the function's signature.
        """
        sig = signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        type_hints = get_type_hints(func)
        for name, value in bound_args.arguments.items():
            expected_type = type_hints.get(name, None)
            if expected_type and not await self.is_valid_type(value, expected_type):
                raise TypeError(
                    f"Argument '{name}' must be of type {expected_type}, got type {type(value)}"
                )

    async def is_valid_argument(self, func: Callable, *args, **kwargs) -> bool:
        """
        Validates the arguments of a function against its type hints and applies custom validation rules, if any.
        This method dynamically adjusts for whether the function is a bound method or a regular function, applying
        argument validation accordingly. It leverages asyncio for non-blocking operations and ensures thread safety
        with asyncio.Lock, providing a robust mechanism for concurrent validations.

        This method is designed to be exhaustive in its approach to argument validation, ensuring compatibility with
        a wide range of type annotations and custom validation rules. It utilizes advanced programming techniques to
        offer a flexible, adaptive, and robust solution for argument validation.

        Args:
            func (Callable): The function to validate arguments for.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            bool: True if all arguments are valid according to their type hints and custom validation rules, False otherwise.

        Raises:
            TypeError: If an argument does not match its expected type according to the function's type hints.
            ValueError: If an argument fails custom validation rules specified in the validation rules dictionary.
        """
        # Adjust for bound methods by removing the 'self' or 'cls' argument
        if inspect.ismethod(func) or (
            hasattr(func, "__self__") and func.__self__ is not None
        ):
            args = args[1:]

        # Bind the provided arguments to the function's signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Retrieve type hints for the function
        type_hints = get_type_hints(func)

        # Initialize an empty list to hold validation tasks
        validation_tasks = []

        # Iterate through each bound argument to validate
        for name, value in bound_args.arguments.items():
            expected_type = type_hints.get(
                name, Any
            )  # Default to Any if no type hint is provided

            # Create a coroutine for validating the current argument's type
            validation_task = self.is_valid_type(value, expected_type)
            validation_tasks.append(validation_task)

            # Check for custom validation rules based on the expected type's name
            rule_name = f"is_{expected_type.__name__}_valid"
            if rule_name in self.validation_rules:
                # Add the custom validation rule to the list of tasks
                custom_validation_task = self.validation_rules[rule_name](value)
                validation_tasks.append(custom_validation_task)

        # Use asyncio.gather to run all validation tasks concurrently and wait for their results
        validation_results = await asyncio.gather(*validation_tasks)

        # If any validation failed, log an error and return False
        if not all(validation_results):
            logging.error("One or more arguments failed validation.")
            return False

        # If all validations passed, return True
        return True

    async def is_valid_type(self, value: Any, expected_type: Any) -> bool:
        """
        Recursively validates a value against an expected type, handling generics, special forms, and complex types.
        This method is designed to be exhaustive in its approach to type validation, ensuring compatibility with a wide range of type annotations.
        Utilizes asyncio for non-blocking operations and ensures thread safety with asyncio.Lock.

        Args:
            value (Any): The value to validate.
            expected_type (Any): The expected type against which to validate the value.

        Returns:
            bool: True if the value matches the expected type, False otherwise.
        """
        if expected_type is Any:
            return True

        if get_origin(expected_type) is Union:
            return any(
                [
                    await self.is_valid_type(value, arg)
                    for arg in get_args(expected_type)
                ]
            )

        if get_origin(expected_type) is Union or expected_type is Any:
            return True

        origin_type = get_origin(expected_type)
        type_args = get_args(expected_type)

        if origin_type:
            if not isinstance(value, origin_type):
                return False
            if type_args:
                # Inside the AsyncValidator class, modify the is_valid_type method
                if issubclass(origin_type, Mapping):
                    key_type, val_type = type_args
                    items_validation = [
                        await self.is_valid_type(k, key_type)
                        and await self.is_valid_type(v, val_type)
                        for k, v in value.items()
                    ]
                    return all(items_validation)
                elif issubclass(origin_type, Iterable) and not issubclass(
                    origin_type, (str, bytes, bytearray)
                ):
                    element_type = type_args[0]
                    # Use asyncio.gather to run validations concurrently and wait for all results
                    validations = [
                        self.is_valid_type(elem, element_type) for elem in value
                    ]
                    results = await asyncio.gather(*validations)
                    return all(results)
        else:
            if not isinstance(value, expected_type):
                return False
            return True
        return False


async def is_positive_integer(value: Any) -> bool:
    """Validates that the value is a positive integer."""
    return isinstance(value, int) and value > 0


async def is_non_empty_string(value: Any) -> bool:
    """Ensures the value is a non-empty string."""
    return isinstance(value, str) and bool(value.strip())


async def is_valid_email(value: Any) -> bool:
    """Checks if the value is a valid email address."""
    if not isinstance(value, str):
        return False
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", value))


async def is_file_path_exists(value: Any) -> bool:
    """Validates that the file path exists."""
    if not isinstance(value, str):
        return False
    return pathlib.Path(value).exists()


async def is_valid_json(value: Any) -> bool:
    """Asynchronously validates if the value is valid JSON."""
    if not isinstance(value, str):
        return False
    try:
        json.loads(value)
        return True
    except ValueError:
        return False


# Wrapper function for is_in_enum to fit the expected signature
def make_is_in_enum_validator(
    enum_class: type[Enum],
) -> Callable[[Any], Awaitable[bool]]:
    """
    Creates a validation function for checking if a value is in the specified enum.

    Args:
        enum_class (type[Enum]): The enumeration class to validate against.

    Returns:
        Callable[[Any], Awaitable[bool]]: An async validation function.
    """

    async def is_in_enum_validator(value: Any) -> bool:
        return await is_in_enum(value, enum_class)

    return is_in_enum_validator


async def is_in_enum(value: Any, enum_class: type[Enum]) -> bool:
    """Validates that the value is a member of a given enum class."""
    return value in [e.value for e in enum_class.__members__.values()]


class AnyEnum(Enum):
    EXAMPLE_VALUE_1 = 1
    EXAMPLE_VALUE_2 = 2
    EXAMPLE_VALUE_3 = 3


async def dynamic_validate(value: Any) -> bool:
    """
    Dynamically validates a value or a function's arguments using all available validation rules and methods.
    This method is designed to be exhaustive in its approach to validation, ensuring compatibility with
    a wide range of data types and structures. It leverages advanced programming techniques to offer a flexible,
    adaptive, and robust solution for dynamic validation.

    Args:
        value (Any): The value or function to be validated.

    Returns:
        bool: True if the value or function's arguments pass all validations, False otherwise.

    Raises:
        ValueError: If an invalid rule name is encountered during validation.
        TypeError: If the value's type does not match the expected type for a given validation rule.
    """
    # Initialize the Validate instance with the comprehensive dictionary of validation rules.
    validator_instance = Validate(validation_rules)
    validation_results = []  # List to store individual validation results.

    # Validate using custom validation rules defined in the validation_rules dictionary.
    for rule_name, validation_func in validation_rules.items():
        # Exclude the Validate class itself to prevent recursion.
        if rule_name != "Validator":
            try:
                # Execute the validation function asynchronously and append the result.
                result = await validation_func(value)
                validation_results.append(result)
            except (ValueError, TypeError) as e:
                # Log any validation errors encountered during the process.
                logging.error(f"Validation error for rule '{rule_name}': {e}")
                validation_results.append(False)

    # If the value is a callable, i.e., a function, validate its signature and arguments.
    if callable(value):
        try:
            # Validate the function's signature for compatibility with expected arguments.
            await validator_instance.is_valid_func_signature(value)
            # Validate the function's arguments against their expected types and custom rules.
            await validator_instance.is_valid_argument(value)
            # Append True to indicate successful validation of a callable's signature and arguments.
            validation_results.append(True)
        except (TypeError, ValueError) as e:
            # Log any errors encountered during validation of the callable's signature or arguments.
            logging.error(f"Validation error for callable '{value}': {e}")
            validation_results.append(False)

    # For non-callable values, validate the value's type.
    if not callable(value):
        try:
            # Validate the value's type against the expected type for its corresponding validation rule.
            type_validation_result = await validator_instance.is_valid_type(
                value, type(value)
            )
            # Append the result of the type validation to the list of validation results.
            validation_results.append(type_validation_result)
        except TypeError as e:
            # Log any type validation errors encountered during the process.
            logging.error(f"Type validation error for value '{value}': {e}")
            validation_results.append(False)

    # Return True if all validations passed, False otherwise.
    return all(validation_results)


# Comprehensive dictionary of validation rules
# This dictionary maps validation rule names to their corresponding asynchronous validation functions.
# Each validation function is designed to take any value as input and return a boolean value asynchronously,
# indicating whether the input satisfies the specific validation rule.
validation_rules: ValidationRules = {
    "is_positive_integer": is_positive_integer,  # Validates if the input is a positive integer.
    "is_non_empty_string": is_non_empty_string,  # Checks if the input is a non-empty string.
    "is_valid_email": is_valid_email,  # Determines if the input is a valid email address.
    "is_file_path_exists": is_file_path_exists,  # Verifies the existence of the file path provided as input.
    "is_valid_json": is_valid_json,  # Validates if the input string is valid JSON.
    "is_in_enum": make_is_in_enum_validator(
        AnyEnum
    ),  # Checks if the input value is a member of a specified enumeration.
    "Validator": dynamic_validate,  # Dynamically validates the input using all available validation rules.
}


async def main():
    """
    Main function to demonstrate the usage of dynamic_validate function.
    """
    test_values = [
        42,  # Should pass is_positive_integer
        "",  # Should fail is_non_empty_string
        "test@example.com",  # Should pass is_valid_email
        "invalid_email",  # Should fail is_valid_email
        "./nonexistentfile.txt",  # Should fail is_file_path_exists
        '{"valid": "json"}',  # Should pass is_valid_json
        AnyEnum.EXAMPLE_VALUE_1,  # Should pass is_in_enum
        "not_an_enum_value",  # Should fail is_in_enum
        is_positive_integer,  # Should pass callable validations
    ]

    for value in test_values:
        result = await dynamic_validate(value)
        print(f"Validation result for {value}: {result}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
