import logging
import logging.config  # needed when logging_config doesn't start with logging.config
from copy import copy
from django.conf import settings
from django.core import mail
from django.core.mail import get_connection
from django.core.management.color import color_style
from django.utils.module_loading import import_string
class ServerFormatter(logging.Formatter):
    default_time_format = '%d/%b/%Y %H:%M:%S'

    def __init__(self, *args, **kwargs):
        self.style = color_style()
        super().__init__(*args, **kwargs)

    def format(self, record):
        msg = record.msg
        status_code = getattr(record, 'status_code', None)
        if status_code:
            if 200 <= status_code < 300:
                msg = self.style.HTTP_SUCCESS(msg)
            elif 100 <= status_code < 200:
                msg = self.style.HTTP_INFO(msg)
            elif status_code == 304:
                msg = self.style.HTTP_NOT_MODIFIED(msg)
            elif 300 <= status_code < 400:
                msg = self.style.HTTP_REDIRECT(msg)
            elif status_code == 404:
                msg = self.style.HTTP_NOT_FOUND(msg)
            elif 400 <= status_code < 500:
                msg = self.style.HTTP_BAD_REQUEST(msg)
            else:
                msg = self.style.HTTP_SERVER_ERROR(msg)
        if self.uses_server_time() and (not hasattr(record, 'server_time')):
            record.server_time = self.formatTime(record, self.datefmt)
        record.msg = msg
        return super().format(record)

    def uses_server_time(self):
        return self._fmt.find('{server_time}') >= 0