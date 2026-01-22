from __future__ import annotations
import inspect
import logging
from tenacity import retry, wait_exponential, stop_after_delay, before_sleep_log, retry_unless_exception_type, retry_if_exception_type, retry_if_exception
from typing import Optional, Union, Tuple, Type, TYPE_CHECKING
class retry_if_type(retry_if_exception):
    """
    Retries if the exception is of the given type
    """

    def __init__(self, exception_types: Union[Type[BaseException], Tuple[Type[BaseException], ...]]=Exception, excluded_types: Union[Type[BaseException], Tuple[Type[BaseException], ...]]=None):
        self.exception_types = exception_types
        self.excluded_types = excluded_types
        super().__init__(lambda e: self.validate_exception(e))

    def validate_exception(self, e: BaseException) -> bool:
        if e.args and e.args[0] == 'PING':
            print('EXCLUDED PING')
            return False
        return isinstance(e, self.exception_types) and (not isinstance(e, self.excluded_types))