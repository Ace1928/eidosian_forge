from typing import (
class PassThroughException(Exception):
    """
    Normally all unhandled exceptions raised during commands get printed to the user.
    This class is used to wrap an exception that should be raised instead of printed.
    """

    def __init__(self, *args: Any, wrapped_ex: BaseException) -> None:
        """
        Initializer for PassThroughException
        :param wrapped_ex: the exception that will be raised
        """
        self.wrapped_ex = wrapped_ex
        super().__init__(*args)