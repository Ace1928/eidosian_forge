from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
from . import errors
from .commands import Command
from .option import Option, RegistryOption
def _parse_version_info_format(format):
    """Convert a string passed by the user into a VersionInfoFormat.

    This looks in the version info format registry, and if the format
    cannot be found, generates a useful error exception.
    """
    try:
        return version_info_formats.get_builder(format)
    except KeyError:
        formats = version_info_formats.get_builder_formats()
        raise errors.CommandError(gettext('No known version info format {0}. Supported types are: {1}').format(format, formats))