from queue import Queue
from typing import TYPE_CHECKING, Optional
Put the new text in the queue. If the stream is ending, also put a stop signal in the queue.