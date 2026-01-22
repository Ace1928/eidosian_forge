from breezy import controldir, errors
from ... import version_info  # noqa: F401
class DarcsDirFormat(controldir.ControlDirFormat):
    """Darcs directory format."""

    def get_converter(self):
        raise NotImplementedError(self.get_converter)

    def get_format_description(self):
        return 'darcs control directory'

    def initialize_on_transport(self, transport):
        raise errors.UninitializableFormat(self)

    def is_supported(self):
        return False

    def supports_transport(self, transport):
        return False

    @classmethod
    def _known_formats(self):
        return {DarcsDirFormat()}

    def check_support_status(self, allow_unsupported, recommend_upgrade=True, basedir=None):
        raise DarcsUnsupportedError()

    def open(self, transport):
        DarcsProber().probe_transport(transport)
        raise NotImplementedError(self.open)