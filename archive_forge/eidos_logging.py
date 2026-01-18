import logging  # 🪵 Imports the 'logging' module for implementing logging functionalities.
from debugpy import (
    configure,
)  # 🐞 Imports 'configure' from 'debugpy' for setting up debugging.
import os
import time
import sys
import logging
import logging.handlers
from logging import (
    LogRecord,
    Formatter,
    StreamHandler,
)  # 🪵 Imports specific logging classes for record handling, formatting, and stream output.
import logging.config
from typing import (
    Optional,
    Union,
    Any,
)  # 🖋️ Imports typing hints for optional values, unions, and any type.
import sys  # 🐍 Imports the 'sys' module for system-specific parameters and functions, like standard output.
import psutil  # 📊 Imports the 'psutil' module for system resource monitoring.

from eidos_resource import (  # 📦 Imports specific functions from the 'eidos_resource' module.
    ResourceUsage,  # 📊 Type hint for resource usage data.
    _get_resource_usage,  # 📊 Function to get current resource usage.
)
from eidos_formatter import (
    EidosFormatter,
)  # 🖋️ Imports the custom formatter for Eidosian logs.
from eidos_profiler import (
    _trace_function,
)  # 🔍 Imports the function for detailed tracing.

from eidos_config import (  # 📦 Imports specific configurations from the 'eidos_config' module.
    LoggingConfig,  # ⚙️ Class for centralized logging configuration.
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
)


def _configure_console_handler(
    logger: logging.Logger,  # 🪵 The logger instance to configure.
    config: LoggingConfig,  # ⚙️ The logging configuration.
) -> None:
    """⚙️ Configures the console logging handler.

    This function sets up a StreamHandler for console output, using the provided logging configuration.
    It creates a custom EidosFormatter, sets the formatter for the handler, and adds the handler to the logger
    if a handler with the same stream does not already exist.

    [all]
        This function configures the console logging handler for the given logger.
        It sets up a StreamHandler, creates a custom EidosFormatter, and adds the handler to the logger.

    Args:
        logger (logging.Logger): 🪵 The logger instance to configure.
        config (LoggingConfig): ⚙️ The logging configuration.

    Returns:
        None: 🚫 This function does not return any value. It configures the logger as a side effect.
    """
    stream_output = (  # 📤 Determines the output stream.
        config.stream_output  # 📤 Uses the configured stream if available.
        if config.stream_output
        is not None  # 📤 Checks if a custom stream is configured.
        else sys.stdout  # 📤 Uses the standard output stream if no custom stream is configured.
    )
    console_handler = StreamHandler(  # 📤 Creates a StreamHandler for console output.
        stream_output  # 📤 Sets the output stream for the handler.
    )
    formatter = EidosFormatter(  # 📝 Creates a custom EidosFormatter for log messages.
        (  # 📝 Uses the configured log format or the default log format.
            config.log_format if config.log_format else DEFAULT_LOG_FORMAT
        ),
        datefmt=config.datetime_format,  # 📅 Sets the datetime format for the formatter.
        use_json=config.log_format_type  # 📝 Sets JSON formatting if specified.
        == "json",
        include_uuid=config.include_uuid,  # 🆔 Includes UUID in log messages if specified.
    )
    console_handler.setFormatter(  # 📝 Sets the formatter for the console handler.
        formatter  # 📝 Applies the custom formatter to the console handler.
    )
    if not any(  # 🔍 Checks if a handler with the same stream already exists.
        handler.stream  # 📤 Gets the stream of the current handler.
        == stream_output  # 📤 Checks if the handler's stream matches the desired output stream.
        for handler in logger.handlers  # 🪵 Iterates through the logger's handlers.
        if isinstance(  # 📤 Checks if the handler is a StreamHandler.
            handler, StreamHandler
        )
    ):
        logger.addHandler(  # 🪵 Adds the console handler to the logger if it doesn't already exist.
            console_handler  # 🪵 Adds the configured console handler to the logger.
        )


def _configure_file_handler(
    logger: logging.Logger,  # 🪵 The logger instance to configure.
    config: LoggingConfig,  # ⚙️ The logging configuration.
    numeric_level: int,  # 🎚️ The numeric log level.
) -> None:
    """⚙️ Configures the file logging handler.

    This function sets up a FileHandler for logging to a file, using the provided logging configuration.
    It determines the file log level, creates a custom EidosFormatter, sets the formatter and level for the handler,
    and adds the handler to the logger if a handler for the same file does not already exist.

    [all]
        This function configures the file logging handler for the given logger.
        It sets up a FileHandler, determines the file log level, creates a custom EidosFormatter,
        and adds the handler to the logger.

    Args:
        logger (logging.Logger): 🪵 The logger instance to configure.
        config (LoggingConfig): ⚙️ The logging configuration.
        numeric_level (int): 🎚️ The numeric log level.

    Returns:
        None: 🚫 This function does not return any value. It configures the logger as a side effect.
    """
    if not config.log_to_file:  # 📁 Checks if file logging is enabled.
        return  # 🚪 Exits if file logging is not enabled.

    file_numeric_level = (  # 🎚️ Initializes the file log level with the console log level.
        numeric_level
    )
    if config.file_log_level:  # 🎚️ Checks if a specific file log level is configured.
        if isinstance(  # 🎚️ Checks if the file log level is a string.
            config.file_log_level, str
        ):
            file_numeric_level = (
                getattr(  # 🎚️ Gets the numeric log level from the string.
                    logging, config.file_log_level.upper(), None
                )
            )
            if not isinstance(  # 🎚️ Checks if the retrieved level is a valid integer.
                file_numeric_level, int
            ):
                logger.error(  # 🪵 Logs an error if the file log level is invalid.
                    f"Invalid file log level: {config.file_log_level}. Using console log level."  # 🪵 Error message.
                )
                file_numeric_level = numeric_level  # 🎚️ Reverts to the console log level if the file log level is invalid.
        elif isinstance(  # 🎚️ Checks if the file log level is an integer.
            config.file_log_level, int
        ):
            file_numeric_level = (  # 🎚️ Uses the provided integer log level.
                config.file_log_level
            )
        else:  # 🎚️ Handles cases where the file log level is of an invalid type.
            logger.error(  # 🪵 Logs an error for an invalid file log level type.
                f"Invalid file log level type: {type(config.file_log_level)}. Using console log level."  # 🪵 Error message.
            )
            file_numeric_level = numeric_level  # 🎚️ Reverts to the console log level if the file log level type is invalid.

    try:  # 🔒 Starts a try block to handle potential errors during file logging setup.
        file_handler = (  # 📁 Creates a file handler for logging to a file.
            logging.FileHandler(
                config.log_to_file,  # 📁 Sets the file path.
                mode="a",  # 📁 Sets the file mode to append.
                encoding="utf-8",  # 📁 Sets the file encoding to UTF-8.
            )
        )
        file_formatter = EidosFormatter(  # 📝 Creates a custom formatter for Eidosian logs.
            (  # 📝 Uses the configured log format or the default log format.
                config.log_format if config.log_format else DEFAULT_LOG_FORMAT
            ),
            datefmt=config.datetime_format,  # 📅 Sets the datetime format for the formatter.
            use_json=config.log_format_type  # 📝 Sets JSON formatting if specified.
            == "json",
            include_uuid=config.include_uuid,  # 🆔 Includes UUID in log messages if specified.
        )
        file_handler.setFormatter(  # 📝 Sets the formatter for the file handler.
            file_formatter  # 📝 Applies the custom formatter to the file handler.
        )
        file_handler.setLevel(  # 🎚️ Sets the log level for the file handler.
            file_numeric_level  # 🎚️ Sets the log level for the file handler.
        )
        if not any(  # 🔍 Checks if a file handler for the same file already exists.
            isinstance(  # 📁 Checks if the handler is a FileHandler.
                handler, logging.FileHandler
            )
            and handler.baseFilename  # 📁 Gets the base filename of the handler.
            == os.path.abspath(  # 📁 Checks if the handler's file matches the configured file.
                config.log_to_file
            )
            for handler in logger.handlers  # 🪵 Iterates through the logger's handlers.
        ):
            logger.addHandler(  # 🪵 Adds the file handler to the logger if it doesn't already exist.
                file_handler  # 🪵 Adds the configured file handler to the logger.
            )
        logger.debug(  # 🪵 Logs a debug message indicating that file logging is enabled.
            f"📝 Logging to file enabled at level: {logging.getLevelName(file_numeric_level)} in: {config.log_to_file}"  # 🪵 Debug message.
        )
    except Exception as e:  # 🔒 Catches any exceptions during file logging setup.
        logger.error(  # 🪵 Logs an error message and disables file logging.
            f"🔥 Error setting up file logging: {e}. File logging disabled."  # 🪵 Error message.
        )


def _configure_detailed_tracing(
    logger: logging.Logger,  # 🪵 The logger instance to configure.
    config: LoggingConfig,  # ⚙️ The logging configuration.
    trace_level: int = 5,  # 🔍 The trace level, defaults to 5.
) -> None:
    """⚙️ Configures detailed tracing if enabled.

    This function sets up detailed tracing by setting a trace function using sys.settrace.
    The trace function calls the _trace_function from the eidos_profiler module, which logs
    detailed information about function calls and variable states.

    [all]
        This function configures detailed tracing for the given logger.
        It sets up a trace function using sys.settrace and calls the _trace_function.

    Args:
        logger (logging.Logger): 🪵 The logger instance to configure.
        config (LoggingConfig): ⚙️ The logging configuration.
        trace_level (int): 🔍 The trace level, defaults to 5.

    Returns:
        None: 🚫 This function does not return any value. It configures detailed tracing as a side effect.
    """
    if not config.detailed_tracing:  # 🔍 Checks if detailed tracing is enabled.
        return  # 🚪 Exits if detailed tracing is not enabled.
    try:  # 🔒 Starts a try block to handle potential errors during tracing setup.
        sys.settrace(  # 🔍 Sets the trace function for the system.
            lambda frame, event, arg: _trace_function(  # 🔍 Defines a lambda function to call the trace function.
                frame,
                event,
                arg,
                logger,
                trace_level,  # 🔍 Passes the frame, event, arguments, logger, and trace level to the trace function.
            )
        )
        logger.debug(  # 🪵 Logs a debug message indicating that detailed tracing is enabled.
            "🔍 Detailed tracing enabled."  # 🪵 Debug message.
        )
    except Exception as e:  # 🔒 Catches any exceptions during tracing setup.
        logger.error(  # 🪵 Logs an error message if tracing setup fails.
            f"🔥 Error enabling detailed tracing: {e}"  # 🪵 Error message.
        )


def _configure_debugpy_trigger(logger: logging.Logger, config: LoggingConfig) -> None:
    """⚙️ Configures debugpy trigger if a level is set.

    This function sets up a custom logging handler that triggers a breakpoint when a log message
    with a level at or above the configured debugpy_trigger_level is emitted. This allows for
    easy debugging by automatically attaching a debugger when a specific log level is reached.

    [all]
        This function configures a debugpy trigger for the given logger.
        It sets up a custom logging handler that triggers a breakpoint when a log message
        with a level at or above the configured debugpy_trigger_level is emitted.

    Args:
        logger (logging.Logger): 🪵 The logger instance to configure.
        config (LoggingConfig): ⚙️ The logging configuration.

    Returns:
        None: 🚫 This function does not return any value. It configures the debugpy trigger as a side effect.
    """
    if (  # 🐞 Checks if a debugpy trigger level is set.
        config.debugpy_trigger_level is None
    ):
        return  # 🚪 Exits if no trigger level is set.

    trigger_level = getattr(  # 🐞 Gets the numeric trigger level from the string.
        logging, str(config.debugpy_trigger_level).upper(), None
    )
    if not isinstance(  # 🐞 Checks if the retrieved level is a valid integer.
        trigger_level, int
    ):
        try:  # 🔒 Starts a try block to handle potential errors during level conversion.
            trigger_level = int(  # 🐞 Tries to convert the trigger level to an integer.
                config.debugpy_trigger_level
            )
        except (  # 🔒 Catches a ValueError if the trigger level cannot be converted to an integer.
            ValueError
        ):
            logger.error(  # 🪵 Logs an error message if the trigger level is invalid.
                f"Invalid debugpy trigger level: {config.debugpy_trigger_level}"  # 🪵 Error message.
            )
            return  # 🚪 Exits if the trigger level is invalid.

    class DebugpyHandler(StreamHandler):  # 🐞 Defines a custom handler for debugpy.
        """🐞 Custom logging handler that triggers a debugpy breakpoint."""

        def emit(  # 🐞 Overrides the emit method to trigger the debugger.
            self, record: LogRecord
        ):
            """🐞 Emits a log record and triggers a breakpoint if the level is high enough.

            Args:
                record (LogRecord): 🪵 The log record to emit.

            Returns:
                None: 🚫 This method does not return any value. It triggers a breakpoint as a side effect.
            """
            if (  # 🐞 Checks if the log record's level is at or above the trigger level.
                record.levelno >= trigger_level
            ):
                configure(  # 🐞 Configures debugpy to wait for a client.
                    wait_for_client=True
                )
                breakpoint()  # 🐞 Triggers a breakpoint to start the debugger.

    debugpy_handler = (  # 🐞 Creates an instance of the custom debugpy handler.
        DebugpyHandler()
    )
    formatter = Formatter(  # 📝 Creates a formatter for the debugpy handler.
        config.log_format if config.log_format else DEFAULT_LOG_FORMAT
    )
    debugpy_handler.setFormatter(  # 📝 Sets the formatter for the debugpy handler.
        formatter
    )
    logger.addHandler(debugpy_handler)  # 🪵 Adds the debugpy handler to the logger.
    logger.debug(  # 🪵 Logs a debug message indicating that the debugpy trigger is enabled.
        f"⚙️ Debugpy trigger enabled for log level: {logging.getLevelName(trigger_level)}."  # 🪵 Debug message.
    )


LLM_TRACE = 5  # 🔬 Define a custom log level for LLM tracing, specifically for detailed LLM operation insights.


class Logger(
    logging.Logger
):  # 🪵 Extends the base Logger class to add custom functionality, enhancing the standard logging capabilities.
    """🪵 Custom Logger class extending logging.Logger with additional functionality.

    This class provides a custom logger with an additional log level for LLM tracing.
    It inherits from the standard logging.Logger class and adds the `llm_trace` method.
    """

    def llm_trace(self, message: str, *args: Any, **kwargs: Any) -> None:
        """🔬 Pinpoints the innermost workings of the LLM. Logs messages at the LLM_TRACE level.

        This method logs messages specifically related to LLM (Large Language Model) tracing.
        It checks if the LLM_TRACE level is enabled for the logger and then logs the message
        at that level.

        Args:
            message (str): 📝 The message to log, providing context about the LLM's operation.
            *args (Any): 📦 Additional positional arguments passed to the logging framework.
            **kwargs (Any): 🔑 Additional keyword arguments passed to the logging framework.

        Returns:
            None: 🚫 This method does not return any value. It performs logging as a side effect.
        """
        if self.isEnabledFor(
            LLM_TRACE
        ):  # 🔬 Checks if the LLM_TRACE level is enabled for this logger instance.
            self._log(
                LLM_TRACE, message, args, **kwargs
            )  # 🔬 Logs the message at the LLM_TRACE level, capturing detailed LLM activity.


logging.setLoggerClass(
    Logger
)  # 🪵 Sets the custom Logger class as the default logger, ensuring all loggers use the extended functionality.
logging.addLevelName(
    LLM_TRACE, "LLM_TRACE"
)  # 🪵 Adds the custom LLM_TRACE level to the logging module, making it available for use in logging configurations.


def configure_logging(
    *args,  # 📦 Accepts positional arguments, the first of which can be the log level (deprecated).
    log_level: Optional[
        Union[str, int]
    ] = None,  # 🎚️ Defines the log level, can be a string (e.g., "DEBUG", "INFO") or an integer (e.g., 10, 20), optional.
    log_format: Optional[
        str
    ] = None,  # 📝 Defines the log format string, optional. Specifies how log messages are formatted.
    log_to_file: Optional[
        str
    ] = None,  # 📁 Defines the path to the log file, optional. If provided, logs will be written to this file.
    file_log_level: Optional[
        Union[str, int]
    ] = None,  # 🎚️ Defines the log level for the file output, optional. Can be a string or an integer.
    detailed_tracing: Optional[
        bool
    ] = None,  # 🔍 Enables or disables detailed tracing, optional. If True, enables detailed tracing of function calls.
    adaptive_logging: Optional[
        bool
    ] = None,  # ⚙️ Enables or disables adaptive logging, optional. If True, enables dynamic adjustment of log levels based on system conditions.
    logger_name: Optional[
        str
    ] = None,  # 🏷️ Defines the name of the logger, optional. Defaults to the name of the current module if not provided.
    stream_output: Optional[
        Any
    ] = None,  # 📤 Defines the output stream, optional. Defaults to sys.stdout if not provided.
    log_format_type: str = "text",  # 📝 Defines the log format type, either 'text' for standard formatting or 'json' for JSON output, defaults to 'text'.
    include_uuid: bool = False,  # 🆔 Includes a UUID in each log record if True, defaults to False. Useful for tracking individual log events.
    datetime_format: Optional[
        str
    ] = None,  # 📅 Defines the datetime format string, optional. If None, uses the default datetime format.
    debugpy_trigger_level: Optional[
        Union[str, int]
    ] = None,  # 🐞 Defines the log level that triggers the debugger, optional. When this level is reached, a breakpoint is triggered.
    adaptive_interval: int = 1,  # ⏱️ Defines the interval for adaptive logging checks in seconds, defaults to 1. How often system resources are checked.
    adaptive_cpu_threshold: float = 80.0,  # 🌡️ Defines the CPU usage threshold for adaptive logging, defaults to 80.0. If CPU usage exceeds this, logging may be adjusted.
    adaptive_mem_threshold: float = 80.0,  # 🧠 Defines the memory usage threshold for adaptive logging, defaults to 80.0. If memory usage exceeds this, logging may be adjusted.
) -> logging.Logger:
    """✨✍️ Configures logging with meticulous Eidosian detail, extensive configurability, and adaptive behavior.

    This function serves as the primary entry point for setting up the Eidosian logging system.
    It allows for extensive customization of logging behavior, including log levels, formats,
    output destinations, and advanced features like adaptive logging and detailed tracing.

    Args:
        *args: 📦 Positional arguments. If provided, the first argument is taken as the log level (deprecated, use log_level instead).
        log_level (Optional[Union[str, int]]): 🎚️ The logging level for console output (e.g., "DEBUG", "INFO", 10, 20). Defaults to the EIDOS_LOG_LEVEL environment variable or DEBUG.
        log_format (Optional[str]): 📝 The format string for log messages when log_format_type is 'text'. Defaults to the EIDOS_LOG_FORMAT environment variable or a detailed default format.
        log_to_file (Optional[str]): 📁 Optional path to a log file. If provided, logs will be written to this file.
        file_log_level (Optional[Union[str, int]]): 🎚️ Optional logging level for the file output. If not provided, defaults to the console log level.
        detailed_tracing (Optional[bool]): 🔍 If True, enables detailed tracing of function calls and variable states.
        adaptive_logging (Optional[bool]): ⚙️ If True, enables dynamic adjustment of log levels based on system conditions.
        logger_name (Optional[str]): 🏷️ The name of the logger. Defaults to __name__.
        stream_output (Optional[Any]): 📤 The stream to output to, defaults to sys.stdout.
        log_format_type (str): 📝 'text' for standard formatting or 'json' for JSON output. Defaults to 'text'.
        include_uuid (bool): 🆔 If True, adds a UUID to each log record. Defaults to False.
        datetime_format (Optional[str]): 📅 Optional string for custom datetime formatting. If None, uses the default.
        debugpy_trigger_level (Optional[Union[str, int]]): 🐞 If set, attaching a debugger and reaching this log level will trigger a breakpoint.
        adaptive_interval (int): ⏱️ Interval in seconds for adaptive logging checks.
        adaptive_cpu_threshold (float): 🌡️ CPU usage percentage threshold for adaptive logging.
        adaptive_mem_threshold (float): 🧠 Memory usage percentage threshold for adaptive logging.

    Returns:
        logging.Logger: 🪵 A configured logging.Logger instance, ready for use.

    Raises:
        ValueError: 🔥 If the provided log level is invalid, such as an unrecognized string or an invalid integer.
    """
    # ⚙️ Creates a LoggingConfig instance with the provided parameters, encapsulating all logging settings.
    config = LoggingConfig(
        log_level=log_level,  # 🎚️ Sets the log level from the provided argument.
        log_format=log_format,  # 📝 Sets the log format from the provided argument.
        log_to_file=log_to_file,  # 📁 Sets the log file path from the provided argument.
        file_log_level=file_log_level,  # 🎚️ Sets the file log level from the provided argument.
        detailed_tracing=detailed_tracing,  # 🔍 Sets detailed tracing from the provided argument.
        adaptive_logging=adaptive_logging,  # ⚙️ Sets adaptive logging from the provided argument.
        logger_name=logger_name,  # 🏷️ Sets the logger name from the provided argument.
        stream_output=stream_output,  # 📤 Sets the output stream from the provided argument.
        log_format_type=log_format_type,  # 📝 Sets the log format type from the provided argument.
        include_uuid=include_uuid,  # 🆔 Sets UUID inclusion from the provided argument.
        datetime_format=datetime_format,  # 📅 Sets the datetime format from the provided argument.
        debugpy_trigger_level=debugpy_trigger_level,  # 🐞 Sets the debugpy trigger level from the provided argument.
        adaptive_interval=adaptive_interval,  # ⏱️ Sets the adaptive interval from the provided argument.
        adaptive_cpu_threshold=adaptive_cpu_threshold,  # 🌡️ Sets the CPU threshold from the provided argument.
        adaptive_mem_threshold=adaptive_mem_threshold,  # 🧠 Sets the memory threshold from the provided argument.
    )

    # 📦 Checks if positional arguments are provided and no log level is set.
    if args and not config.log_level:
        # 📦 Sets the log level from the first positional argument (deprecated).
        config.log_level = args[0]
    # 🎚️ Checks if the log level is still not set.
    if not config.log_level:
        # 🎚️ Sets the log level to the default EIDOS log level.
        config.log_level = DEFAULT_LOG_LEVEL

    # 🏷️ Sets the logger name or defaults to the current module name.
    logger_name = config.logger_name if config.logger_name else __name__
    # 🪵 Gets or creates a logger with the specified name.
    logger = logging.getLogger(logger_name)
    # 🪵 Prevents log messages from propagating to parent loggers, ensuring they are handled only by this logger.
    logger.propagate = False

    # 🎚️ Declares a variable to hold the numeric log level.
    numeric_level: int
    # 🎚️ Checks if the log level is a string.
    if isinstance(config.log_level, str):
        # 🎚️ Gets the numeric log level from the string, defaulting to 5 (WARNING) if not found.
        numeric_level = getattr(logging, config.log_level.upper(), 5)
        # 🎚️ Checks if the retrieved level is a valid integer.
        if numeric_level is None:
            # 🔥 Raises a ValueError if the log level is invalid.
            raise ValueError(
                f"🔥 Invalid log level: {config.log_level}. Please use a valid level like DEBUG, INFO, WARNING, ERROR, or CRITICAL."
            )  # 🔥 Error message.
    # 🎚️ Checks if the log level is an integer.
    elif isinstance(config.log_level, int):
        # 🎚️ Uses the provided integer log level.
        numeric_level = config.log_level
    # 🎚️ Handles cases where the log level is of an invalid type.
    else:
        # 🎚️ Defaults to the DEBUG log level.
        numeric_level = logging.DEBUG

    # 🪵 Sets the log level for the logger.
    logger.setLevel(numeric_level)

    # ⚙️ Configures the console handler.
    _configure_console_handler(logger, config)
    # ⚙️ Configures the file handler.
    _configure_file_handler(logger, config, numeric_level)
    # ⚙️ Configures detailed tracing.
    _configure_detailed_tracing(logger, config)
    # ⚙️ Configures the debugpy trigger.
    _configure_debugpy_trigger(logger, config)

    # 🎚️ Gets the name of the log level.
    log_level_name = logging.getLevelName(numeric_level)
    # 📝 Sets the display format for the log format.
    log_format_display = (
        "JSON"  # 📝 Sets the display to JSON if the format type is JSON.
        if config.log_format_type == "json"  # 📝 Checks if the format type is JSON.
        else f"'{config.log_format if config.log_format else DEFAULT_LOG_FORMAT}'"  # 📝 Sets the display to the log format string.
    )
    # 📊 Gets the current resource usage.
    resource_usage = _get_resource_usage()
    # 🪵 Logs a debug message indicating that logging is configured.
    logger.debug(
        f"✅ Logging configured at level: {log_level_name} with format: {log_format_display}. Eidosian logging is active. Current Resource Usage: {resource_usage}"
    )  # 🪵 Debug message.
    # 🪵 Returns the configured logger instance.
    return logger


# 🪵 Configures the logger and assigns it to the 'logger' variable, making it globally accessible.
logger: logging.Logger = configure_logging()


def get_logger(name: str) -> logging.Logger:
    """🪵 Retrieves a logger instance by name.

    This function provides a way to get a specific logger instance,
    allowing for modular logging configurations within different parts of the application.

    Args:
        name (str): 🏷️ The name of the logger to retrieve.

    Returns:
        logging.Logger: 🪵 The logger instance associated with the given name.
    """
    # 🪵 Returns the logger instance.
    return logging.getLogger(name)


if __name__ == "__main__":
    """🧪 Main section for testing the logging configuration.

    This section demonstrates how to use the configure_logging and get_logger functions.
    It sets up a basic logging configuration and logs a few messages at different levels.
    """
    # 🪵 Configures the logger with a specific name and log level.
    test_logger = configure_logging(
        logger_name="test_logger",
        log_level="DEBUG",
        log_to_file="test.log",
        detailed_tracing=True,
        adaptive_logging=True,
    )
    # 🪵 Logs a debug message.
    test_logger.debug("🐛 This is a debug message from the test logger.")
    # 🪵 Logs an info message.
    test_logger.info("ℹ️ This is an info message from the test logger.")
    # 🪵 Logs a warning message.
    test_logger.warning("⚠️ This is a warning message from the test logger.")
    # 🪵 Logs an error message.
    test_logger.error("🚨 This is an error message from the test logger.")
    # 🪵 Logs a critical message.
    test_logger.critical("🔥 This is a critical message from the test logger.")

    # 🪵 Retrieves a logger instance by name.
    another_logger = get_logger("another_logger")
    # 🪵 Logs a message using the retrieved logger.
    another_logger.info("ℹ️ This is a message from another logger.")
    # 🪵 Logs a message using the default logger.
    logger.info("ℹ️ This is a message from the default logger.")
