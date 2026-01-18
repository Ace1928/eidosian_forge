import asyncio
import aiofiles
import logging
from typing import Optional, Literal, Union, Any

        Overrides the emit method to write logs asynchronously.

        This method formats the log record and schedules the aio_write coroutine to run
        in the event loop, ensuring that the log message is written asynchronously.

        Args:
            record (logging.LogRecord): The log record to emit.
        