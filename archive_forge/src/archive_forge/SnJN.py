import asyncio
import aiofiles
import logging
from typing import Optional


class AsyncFileHandler(logging.Handler):
    """
    An asynchronous logging handler leveraging aiofiles for non-blocking file writing operations.

    This class extends the logging.Handler to provide asynchronous logging capabilities, ensuring
    that logging operations do not block the main execution flow of an application. It utilizes
    the aiofiles library to perform file I/O operations asynchronously.

    Attributes:
        filename (str): The path to the log file.
        mode (str): The mode in which the log file is opened, defaulting to 'a' (append mode).
        loop (Optional[asyncio.AbstractEventLoop]): The event loop in which asynchronous operations
            will be scheduled. If None, the current running event loop is used.
        aiofile (Optional[aiofiles.base.AiofilesContextManager]): The file object used for asynchronous
            I/O operations. Initialized to None and set during the first write operation.
        lock (asyncio.Lock): An asyncio lock to ensure thread-safe access to the file object.
    """

    def __init__(
        self,
        filename: str,
        mode: str = "a",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """
        Initializes an instance of AsyncFileHandler.

        Args:
            filename (str): The path to the log file.
            mode (str): The mode in which the log file is opened. Defaults to 'a'.
            loop (Optional[asyncio.AbstractEventLoop]): The event loop to use for asynchronous operations.
                If None, the current running event loop is used.
        """
        super().__init__()
        self.filename = filename
        self.mode = mode
        self.loop = loop or asyncio.get_event_loop()
        self.aiofile = None  # Deferred initialization to the aio_open method.
        self.lock: Optional[asyncio.Lock] = None
        self.lock = (
            asyncio.Lock()
        )  # Ensures exclusive access to the file during write operations.

    async def aio_open(self):
        """
        Asynchronously opens the log file for writing.

        This method is responsible for opening the file asynchronously and is called
        before the first write operation to ensure the file is ready for logging.
        """
        if self.lock is None:
            self.lock = asyncio.Lock()
        async with self.lock:
            if self.aiofile is None:
                self.aiofile = await aiofiles.open(self.filename, mode=self.mode)

    async def aio_write(self, msg: str):
        """
        Asynchronously writes a log message to the file.

        Args:
            msg (str): The log message to be written to the file.
        """
        async with self.lock:
            if self.aiofile is None:
                await self.aio_open()
            await self.aiofile.write(msg)
            await self.aiofile.flush()  # Ensures that the message is immediately written to disk.

    def emit(self, record: logging.LogRecord):
        """
        Overrides the emit method to write logs asynchronously.

        This method formats the log record and schedules the aio_write coroutine to run
        in the event loop, ensuring that the log message is written asynchronously.

        Args:
            record (logging.LogRecord): The log record to emit.
        """
        msg = self.format(record)
        asyncio.run_coroutine_threadsafe(self.aio_write(msg + "\n"), self.loop)
