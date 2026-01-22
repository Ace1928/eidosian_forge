import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
class ChangelogCreateError(Exception):
    """Indicates that changelog could not be created, as all the information
    required was not given"""