from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
from . import errors
from .commands import Command
from .option import Option, RegistryOption
Convert a string passed by the user into a VersionInfoFormat.

    This looks in the version info format registry, and if the format
    cannot be found, generates a useful error exception.
    