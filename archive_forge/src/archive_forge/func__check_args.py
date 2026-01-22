import getpass
import argparse
import sys
from . import core
from . import backend
from . import completion
from . import set_keyring, get_password, set_password, delete_password
from .util import platform_
def _check_args(self):
    if self.operation:
        if self.service is None or self.username is None:
            self.parser.error(f'{self.operation} requires service and username')