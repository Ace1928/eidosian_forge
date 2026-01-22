import base64
import binascii
import logging
import bcrypt
import webob
from oslo_config import cfg
from oslo_middleware import base
class ConfigInvalid(Exception):

    def __init__(self, error_msg):
        super().__init__('Invalid configuration file. %(error_msg)s')