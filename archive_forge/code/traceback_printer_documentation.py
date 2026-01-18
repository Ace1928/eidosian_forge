import os
from typing import Optional, Tuple, Union
from .util import NO_UTF8, color, supports_ansi
Output custom formatted tracebacks and errors.

        title (str): The message title.
        *texts (str): The texts to print (one per line).
        RETURNS (str): The formatted traceback. Can be printed or raised
            by custom exception.
        