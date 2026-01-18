import sys
import threading
from typing import Callable, List, Sequence, TextIO
from absl import logging
Returns a message callback that logs messages to a list.

    Args:
      sink: The list to append messages to.

    Returns:
      A function matching the expected signature for message callbacks.
    