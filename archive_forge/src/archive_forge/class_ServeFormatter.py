import copy
import json
import logging
import os
from typing import Optional, Tuple
import ray
from ray.serve._private.common import ServeComponentType
from ray.serve._private.constants import (
from ray.serve.schema import EncodingType, LoggingConfig
class ServeFormatter(logging.Formatter):
    """Serve Logging Formatter

    The formatter will generate the log format on the fly based on the field of record.
    """
    COMPONENT_LOG_FMT = f'%({SERVE_LOG_LEVEL_NAME})s %({SERVE_LOG_TIME})s {{{SERVE_LOG_COMPONENT}}} {{{SERVE_LOG_COMPONENT_ID}}} '

    def __init__(self, component_name: str, component_id: str):
        self.component_log_fmt = ServeFormatter.COMPONENT_LOG_FMT.format(component_name=component_name, component_id=component_id)

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record into the format string.

        Args:
            record: The log record to be formatted.

            Returns:
                The formatted log record in string format.
        """
        record_format = self.component_log_fmt
        record_formats_attrs = []
        if SERVE_LOG_REQUEST_ID in record.__dict__:
            record_formats_attrs.append(SERVE_LOG_RECORD_FORMAT[SERVE_LOG_REQUEST_ID])
        if SERVE_LOG_ROUTE in record.__dict__:
            record_formats_attrs.append(SERVE_LOG_RECORD_FORMAT[SERVE_LOG_ROUTE])
        record_formats_attrs.append(SERVE_LOG_RECORD_FORMAT[SERVE_LOG_MESSAGE])
        record_format += ' '.join(record_formats_attrs)
        formatter = logging.Formatter(record_format)
        return formatter.format(record)