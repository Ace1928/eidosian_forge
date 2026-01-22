import logging
from typing import Any, cast
class VerboseLogger(logging.Logger):
    """Custom Logger, defining a verbose log-level

    VERBOSE is between INFO and DEBUG.
    """

    def verbose(self, msg: str, *args: Any, **kwargs: Any) -> None:
        return self.log(VERBOSE, msg, *args, **kwargs)