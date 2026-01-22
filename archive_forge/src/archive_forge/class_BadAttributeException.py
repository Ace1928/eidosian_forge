import json
from typing import Any, Dict, List, Optional, Tuple, Type, Union
class BadAttributeException(Exception):
    """
    Exception raised when Github returns an attribute with the wrong type.
    """

    def __init__(self, actualValue: Any, expectedType: Union[Dict[Tuple[Type[str], Type[str]], Type[dict]], Tuple[Type[str], Type[str]], List[Type[dict]], List[Tuple[Type[str], Type[str]]]], transformationException: Optional[Exception]):
        self.__actualValue = actualValue
        self.__expectedType = expectedType
        self.__transformationException = transformationException

    @property
    def actual_value(self) -> Any:
        """
        The value returned by Github
        """
        return self.__actualValue

    @property
    def expected_type(self) -> Union[List[Type[dict]], Tuple[Type[str], Type[str]], Dict[Tuple[Type[str], Type[str]], Type[dict]], List[Tuple[Type[str], Type[str]]]]:
        """
        The type PyGithub expected
        """
        return self.__expectedType

    @property
    def transformation_exception(self) -> Optional[Exception]:
        """
        The exception raised when PyGithub tried to parse the value
        """
        return self.__transformationException