class AsyncFileHandler(logging.Handler):
    """

    An asynchronous logging handler using aiofiles for non-blocking file writing.

    """

    def __init__(
        self,
        filename: str,
        mode: str = "a",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """

        Initializes the AsyncFileHandler.

        Args:

            filename (str): The name of the file to which logs are written.

            mode (str): The mode in which the file is opened.

            loop (Optional[asyncio.AbstractEventLoop]): The asyncio event loop. If not provided, the current running loop is used.

        """

        super().__init__()

        self.filename = filename

        self.mode = mode

        self.loop = loop or asyncio.get_event_loop()

        self.aiofile = None  # Initialization moved to aio_open due to async nature

        self.lock = asyncio.Lock()  # Ensure thread safety for file operations

    async def aio_open(self):
        """

        Asynchronously opens the file with aiofiles.

        """

        async with self.lock:

            if self.aiofile is None:

                self.aiofile = await aiofiles.open(self.filename, self.mode)

    async def aio_write(self, msg: str):
        """

        Asynchronously writes a log message to the file.

        Args:

            msg (str): The log message to write.

        """

        async with self.lock:

            if self.aiofile is None:

                await self.aio_open()

            await self.aiofile.write(msg)

            await self.aiofile.flush()

    def emit(self, record):
        """

        Overrides the emit method to write logs asynchronously.

        Args:

            record (logging.LogRecord): The log record to emit.

        """

        msg = self.format(record)

        asyncio.run_coroutine_threadsafe(self.aio_write(msg + "\n"), self.loop)
