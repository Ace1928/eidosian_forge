from typing import Any, Optional, Tuple
from ray.util.annotations import DeveloperAPI
Args:
        category: A short (<20 chars) label for the error.
        description: A longer, human readable description of the error.
        src_exc_info: The source exception info if applicable. This is a
              tuple of (type, exception, traceback) as returned by
              sys.exc_info()

        