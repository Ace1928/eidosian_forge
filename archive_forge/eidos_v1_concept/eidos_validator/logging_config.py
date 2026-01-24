"""
Logging Configuration
--------------------
Configures logging for the Eidos Validator system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def configure_logging(
    level: int = logging.INFO, log_file: Optional[str] = "eidos_validator.log"
) -> logging.Logger:
    """
    Configure logging with detailed format.

    Args:
        level: The logging level (default: INFO)
        log_file: The path to the log file (default: 'eidos_validator.log')
            If None, only console logging is used.

    Returns:
        Logger: Configured logger instance
    """
    logger = logging.getLogger("eidos_validator")

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (IOError, PermissionError) as e:
            logger.warning(f"Failed to create log file: {e}")

    return logger
