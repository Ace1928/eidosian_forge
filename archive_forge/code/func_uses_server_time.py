import logging
import logging.config  # needed when logging_config doesn't start with logging.config
from copy import copy
from django.conf import settings
from django.core import mail
from django.core.mail import get_connection
from django.core.management.color import color_style
from django.utils.module_loading import import_string
def uses_server_time(self):
    return self._fmt.find('{server_time}') >= 0