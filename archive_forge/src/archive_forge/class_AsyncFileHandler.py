import asyncio
import aiofiles
import logging
from typing import Optional, Literal, Union, Any
class AsyncFileHandler(logging.Handler):
    """
    An asynchronous logging handler leveraging aiofiles for non-blocking file writing operations.

    This class extends the logging.Handler to provide asynchronous logging capabilities, ensuring
    that logging operations do not block the main execution flow of an application. It utilizes
    the aiofiles library to perform file I/O operations asynchronously.

    Attributes:
        filename (str): The path to the log file.
        mode (Literal["a", "w"]): The mode in which the log file is opened, defaulting to 'a' (append mode).
        loop (Optional[asyncio.AbstractEventLoop]): The event loop in which asynchronous operations
            will be scheduled. If None, the current running event loop is used.
        aiofile (Optional[aiofiles.AsyncFile]): The file object used for asynchronous
            I/O operations. Initialized to None and set during the first write operation.
        lock (asyncio.Lock): An asyncio lock to ensure thread-safe access to the file object.
    """

    def __init__(self, filename: str, mode: Literal['a', 'w']='a', loop: Optional[asyncio.AbstractEventLoop]=None):
        """
        Initializes an instance of AsyncFileHandler.

        Args:
            filename (str): The path to the log file.
            mode (Literal["a", "w"]): The mode in which the log file is opened. Defaults to 'a'.
            loop (Optional[asyncio.AbstractEventLoop]): The event loop to use for asynchronous operations.
                If None, the current running event loop is used.
        """
        super().__init__()
        self.filename: str = filename
        self.mode: Literal['a', 'w'] = mode
        self.loop: Optional[asyncio.AbstractEventLoop] = loop or asyncio.get_event_loop()
        self.aiofile: Optional[aiofiles.AsyncFile] = None
        self.lock: asyncio.Lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    @log_function_entry_and_exit(logger=logging.getLogger(__name__))
    async def aio_open(self):
        """
        Asynchronously opens the log file for writing.

        This method is responsible for opening the file asynchronously and is called
        before the first write operation to ensure the file is ready for logging.
        """
        if self.aiofile is None:
            async with self.lock:
                if self.aiofile is None:
                    self.aiofile = await aiofiles.open(self.filename, mode=self.mode)
                    self.logger.debug(f'Log file {self.filename} opened in mode {self.mode}.')

    @log_function_entry_and_exit(logger=logging.getLogger(__name__))
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
            await self.aiofile.flush()
            self.logger.debug(f'Log message written to {self.filename}: {msg}')

    @log_function_entry_and_exit(logger=logging.getLogger(__name__))
    def emit(self, record: logging.LogRecord):
        """
        Overrides the emit method to write logs asynchronously.

        This method formats the log record and schedules the aio_write coroutine to run
        in the event loop, ensuring that the log message is written asynchronously.

        Args:
            record (logging.LogRecord): The log record to emit.
        """
        msg = self.format(record)
        if self.loop is None:
            raise ValueError('Event loop is not available for AsyncFileHandler.')
        asyncio.run_coroutine_threadsafe(self.aio_write(msg + '\n'), self.loop)
        self.logger.debug(f'Log record emitted: {record}')