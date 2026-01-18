class ASCIIFormatter:
    """ """

    def __init__(
        self,
        log: Optional[LoggerConfig.LogEntryDictType] = None,
    ) -> None:
        if log is None:
            log = {
                "timestamp": None,  # Placeholder for actual timestamp
                "message": None,  # Placeholder for actual message
                "level": None,  # Placeholder for actual level
                "module": None,  # Placeholder for actual module
                "function": None,  # Placeholder for actual function
                "line": None,  # Placeholder for actual line
                "exc_info": None,  # Placeholder for actual exception info
            }
        """
        Initializes the AdvancedASCIIFormatter class with the default ASCII art components.
        """
        self.log = log
        self.ENTRY_SEPARATOR: str = "-" * 80
        self.TOP_LEFT_CORNER: str = "┌"  # For Message Box Outline
        self.TOP_RIGHT_CORNER: str = "┐"  # For Message Box Outline
        self.BOTTOM_LEFT_CORNER: str = "└"  # For Message Box Outline
        self.BOTTOM_RIGHT_CORNER: str = "┘"  # For Message Box Outline
        self.HORIZONTAL_LINE: str = "─"  # For Message Box Outline
        self.VERTICAL_LINE: str = "│"  # For Message Box Outline
        self.HORIZONAL_DIVIDER_LEFT: str = (
            "├"  # For Info Box Outline Inside Message Box To Organise Entry
        )
        self.HORIZONAL_DIVIDER_RIGHT: str = (
            "┤"  # For Info Box Outline Inside Message Box To Organise Entry
        )
        self.HORIZONTAL_DIVIDER_MIDDLE: str = (
            "┼"  # For Info Box Outline Inside Message Box To Organise Entry
        )
        self.VERTICAL_DIVIDER: str = (
            "┼"  # For Info Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_TOP_LEFT: str = (
            "╔"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_BOTTOM_LEFT: str = (
            "╚"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_TOP_RIGHT: str = (
            "╗"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_BOTTOM_RIGHT: str = (
            "╝"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_LEFT: str = (
            "╠"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_RIGHT: str = (
            "╣"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_MIDDLE: str = (
            "╬"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_VERTICAL: str = (
            "║"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        self.ERROR_INDICATOR_BOX_HORIZONTAL: str = (
            "═"  # For Error Box Outline Inside Message Box To Organise Entry
        )
        # Level symbols for different log levels
        self.LEVEL_SYMBOLS: Dict[LoggerConfig.LogEntryLevelType, str] = {
            LoggerConfig.LogEntryLevelType.NOTSET: "🤷",
            LoggerConfig.LogEntryLevelType.DEBUG: "🐞",
            LoggerConfig.LogEntryLevelType.INFO: "ℹ️",
            LoggerConfig.LogEntryLevelType.WARNING: "⚠️",
            LoggerConfig.LogEntryLevelType.ERROR: "❌",
            LoggerConfig.LogEntryLevelType.CRITICAL: "🚨",
        }
        # Exception Symbols for different exception types
        self.EXCEPTION_SYMBOLS: Dict[Type[LoggerConfig.LogEntryExceptionType], str] = {
            ValueError: "#❌#",  # Value Error
            TypeError: "T❌T",  # Type Error
            KeyError: "K❌K",  # Key Error
            IndexError: "I❌I",  # Index Error
            AttributeError: "A❌A",  # Attribute Error
            Exception: "E❌E",  # General Exception
        }

    def __call__(
        self, log: LoggerConfig.LogEntryDictType
    ) -> LoggerConfig.LogEntryDictType:
        """
        Prepares a log entry dictionary with advanced ASCII formatting for further processing.

        Args:
            log (LoggerConfig.LogEntryDictType): The log entry dictionary to process.

        Returns:
            LoggerConfig.LogEntryDictType: The processed log entry dictionary with enhanced formatting.
        """
        # Initialize a dictionary to hold the formatted log entry components
        formatted_log = {}

        # Format a header for marking clearly the start of the log entry
        formatted_log["header"] = self.ENTRY_SEPARATOR

        # Format the timestamp component of the log entry
        formatted_log["timestamp"] = (
            f"{self.TOP_LEFT_CORNER} {log['timestamp']} {self.TOP_RIGHT_CORNER}"
        )

        # Format the level component of the log entry
        formatted_log["level"] = (
            f"{self.LEVEL_SYMBOLS[log['level']]} {log['level'].name.upper()}"
        )

        # Format the message component of the log entry
        formatted_log["message"] = f"{self.VERTICAL_LINE} {log['message']}"

        # Format the module component of the log entry
        formatted_log["module"] = (
            f"{self.HORIZONAL_DIVIDER_LEFT} Module: {log['module']} {self.HORIZONAL_DIVIDER_RIGHT}"
        )

        # Format the function component of the log entry
        formatted_log["function"] = (
            f"{self.HORIZONTAL_DIVIDER_MIDDLE} Function: {log['function']} {self.HORIZONTAL_DIVIDER_MIDDLE}"
        )

        # Format the line component of the log entry
        formatted_log["line"] = f"{self.VERTICAL_LINE} Line: {log['line']}"

        # Format the exception component of the log entry
        formatted_log["exc_info"] = (
            f"{self.ERROR_INDICATOR_BOX_TOP_LEFT} Exception: {log['exc_info']} {self.ERROR_INDICATOR_BOX_TOP_RIGHT}"
        )

        # Add In the Authorship Details and a Footer Section to act as a clear separator and identifier for the log entry end
        formatted_log["authorship"] = (
            f"{self.BOTTOM_LEFT_CORNER} Author: Lloyd Russell {self.BOTTOM_RIGHT_CORNER}"
        )

        # Combine the exception and authorship components
        formatted_log["exc_info"] = [
            formatted_log["exc_info"] + "\n" + formatted_log["authorship"]
        ]

        # Return the processed log entry dictionary with enhanced formatting
        return {
            "timestamp": formatted_log["timestamp"],
            "message": formatted_log["message"],
            "level": formatted_log["level"],
            "module": formatted_log["module"],
            "function": formatted_log["function"],
            "line": formatted_log["line"],
            "exc_info": formatted_log["exc_info"],
        }

    def format(
        self, log: LoggerConfig.LogEntryDictType
    ) -> LoggerConfig.LogEntryDictType:
        """ """
        # If the log entry is a string, process it as a message and split it into parts
        if isinstance(log, str):
            # Split the message into parts and process each part
            parts = self.split_message(log)
            # Initialize a dictionary to hold the formatted log entry components
            dict: LoggerConfig.LogEntryDictType = {}
            # Process each part of the message
            for part in parts:
                # Process the part and update the dictionary with the formatted components
                dict.update(self(part))
            # Return the dictionary containing the formatted log entry components
            return dict
        # If the log entry is a dictionary, process it as a log entry
        elif isinstance(log, dict):
            # Process the log entry and return the formatted dictionary
            return self(log)
        # If the log entry is neither a string nor a dictionary, raise a TypeError
        else:
            raise TypeError("Log entry must be a string or a dictionary")


class ColoredFormatter:
    """ """

    def __init__(self, log: LoggerConfig.LogEntryDictType) -> None:
        """ """
        self.COLORS: Dict[LoggerConfig.LogLevel, str] = {
            LoggerConfig.LogLevel.DEBUG: "\033[94m",
            LoggerConfig.LogLevel.INFO: "\033[92m",
            LoggerConfig.LogLevel.WARNING: "\033[93m",
            LoggerConfig.LogLevel.ERROR: "\033[91m",
            LoggerConfig.LogLevel.CRITICAL: "\033[95m",
        }
        self.RESET_COLOR: str = "\033[0m"
        self.log = log

    def __call__(
        self, log_entry: LoggerConfig.LogEntryDictType
    ) -> LoggerConfig.LogEntryDictType:
        """
        Formats a log entry with color based on its log level.

        Args:
            log_entry (Union[LogEntry, CollapsibleLogEntry]): The log entry to format.

        Returns:
            str: The formatted log entry with color.
        """
        color: str = self.COLORS.get(log.level, "")
        return super().__call__(
            f"{color}{log.timestamp} | {log.level.name.upper()} | {log.message}{self.RESET_COLOR}"
        )
