import logging
import json
import uuid
from typing import Optional
from logging import LogRecord, Formatter


class EidosFormatter(Formatter):
    """✍️ Custom formatter for Eidosian logs, supporting JSON and UUID."""

    def __init__(
        self,
        format_string: str,
        datefmt: Optional[str],
        use_json: bool,
        include_uuid: bool,
    ):
        super().__init__(format_string, datefmt)
        self.use_json = use_json
        self.include_uuid = include_uuid

    def format(self, record: LogRecord) -> str:
        log_record = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "filename": record.filename,
            "lineno": record.lineno,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
            "uuid": str(uuid.uuid4()),
        }
        if self.use_json:
            return json.dumps(log_record)
        else:
            return super().format(record)
