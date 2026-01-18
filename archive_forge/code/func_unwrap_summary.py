import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def unwrap_summary(summary):
    """Return summary with newlines removed in preparation for wrapping."""
    return re.sub('\\s*\\n\\s*', ' ', summary)