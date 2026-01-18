import errno
import socket
from celery import bootsteps
from celery.exceptions import WorkerLostError
from celery.utils.log import get_logger
from . import state
Fallback blocking event loop for transports that doesn't support AIO.