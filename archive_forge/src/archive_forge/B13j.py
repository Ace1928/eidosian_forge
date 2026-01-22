"""
**1.4 Logger (`logger.py`):**
- **Purpose:** Manages the logging of all module operations with precision and detail.
- **Functions:**
  - `log_message_with_detailed_context(message, level)`: Logs a message at the specified level (DEBUG, INFO, WARNING, ERROR, CRITICAL) with comprehensive details.
"""

import os
import logging
from logging.handlers import RotatingFileHandler


class ComprehensiveLoggingSystem:
    """
    A comprehensive logging system meticulously designed to handle logs across various severity levels with high precision and detail. This class incorporates file rotation,
    custom formatting, and systematic record-keeping to ensure that all log entries are meticulously recorded and easily traceable.

    Attributes:
        logger_instance (logging.Logger): The logger instance used for logging messages.
        log_file_path (str): Full path to the log file where logs are stored.
        max_log_size_bytes (int): Maximum size in bytes before log rotation is triggered.
        backup_logs_count (int): Number of backup log files to retain.
    """

    def __init__(
        self,
        logger_name: str = "ComprehensiveScriptLogger",
        log_directory: str = "logs",
        log_filename: str = "comprehensive_script.log",
        max_log_size: int = 10485760,  # 10MB
        backup_count: int = 5,
    ):
        """
        Initializes the ComprehensiveLoggingSystem with a rotating file handler to manage log file size and backup, ensuring detailed and comprehensive logging.

        Parameters:
            logger_name (str): Name of the logger, defaults to 'ComprehensiveScriptLogger'.
            log_directory (str): Directory where the log file is stored, defaults to 'logs'.
            log_filename (str): Name of the log file, defaults to 'comprehensive_script.log'.
            max_log_size (int): Maximum size of the log file in bytes before rotation, defaults to 10MB.
            backup_count (int): Number of backup log files to maintain, defaults to 5.
        """
        # Ensure the directory for the log file exists
        self.log_file_path = os.path.join(log_directory, log_filename)
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

        # Create and configure logger
        self.logger_instance = logging.getLogger(logger_name)
        self.logger_instance.setLevel(
            logging.DEBUG
        )  # Capture all types of log messages

        # Create a rotating file handler
        handler = RotatingFileHandler(
            self.log_file_path, maxBytes=max_log_size, backupCount=backup_count
        )

        # Define the log format with maximum detail
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger_instance.addHandler(handler)

    def log_message_with_detailed_context(self, message: str, severity_level: str):
        """
        Logs a message at the specified logging level with utmost precision and detail, ensuring all relevant information is captured.

        Parameters:
            message (str): The message to log, detailed and specific to the context.
            severity_level (str): The severity level at which to log the message. Expected values include 'debug', 'info', 'warning', 'error', 'critical'.

        Raises:
            ValueError: If the logging level is not recognized, ensuring strict adherence to logging standards.
        """
        # Validate and convert the severity level to a valid logging method
        log_method = getattr(self.logger_instance, severity_level.lower(), None)
        if log_method is None:
            raise ValueError(
                f"Logging level '{severity_level}' is not valid. Use 'debug', 'info', 'warning', 'error', or 'critical'."
            )
        log_method(message)  # Log the message with detailed context and precision
