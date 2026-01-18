import logging
import socket
from re import sub
from gunicorn.glogging import Logger
Measure request duration
        request_time is a datetime.timedelta
        