from typing import Optional
from typing import Sequence
class WebDriverException(Exception):
    """Base webdriver exception."""

    def __init__(self, msg: Optional[str]=None, screen: Optional[str]=None, stacktrace: Optional[Sequence[str]]=None) -> None:
        super().__init__()
        self.msg = msg
        self.screen = screen
        self.stacktrace = stacktrace

    def __str__(self) -> str:
        exception_msg = f'Message: {self.msg}\n'
        if self.screen:
            exception_msg += 'Screenshot: available via screen\n'
        if self.stacktrace:
            stacktrace = '\n'.join(self.stacktrace)
            exception_msg += f'Stacktrace:\n{stacktrace}'
        return exception_msg