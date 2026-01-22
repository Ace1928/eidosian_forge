import getpass
import argparse
import sys
from . import core
from . import backend
from . import completion
from . import set_keyring, get_password, set_password, delete_password
from .util import platform_
def _load_spec_backend(self):
    if self.keyring_backend is None:
        return
    try:
        if self.keyring_path:
            sys.path.insert(0, self.keyring_path)
        set_keyring(core.load_keyring(self.keyring_backend))
    except (Exception,) as exc:
        self.parser.error(f'Unable to load specified keyring: {exc}')