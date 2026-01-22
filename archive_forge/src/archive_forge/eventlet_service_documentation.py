import socket
import sys
import time
import eventlet.wsgi
import greenlet
from oslo_config import cfg
from oslo_service import service
Start a WSGI server with a new green thread pool.