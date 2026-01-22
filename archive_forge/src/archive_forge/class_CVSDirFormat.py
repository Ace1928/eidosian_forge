from ... import version_info  # noqa: F401
from ... import controldir, errors
from ...transport import register_transport_proto
class CVSDirFormat(controldir.ControlDirFormat):
    """The CVS directory control format."""

    def get_converter(self):
        raise NotImplementedError(self.get_converter)

    def get_format_description(self):
        return 'CVS control directory.'

    def initialize_on_transport(self, transport):
        raise errors.UninitializableFormat(self)

    def is_supported(self):
        return False

    def supports_transport(self, transport):
        return False

    def check_support_status(self, allow_unsupported, recommend_upgrade=True, basedir=None):
        raise CVSUnsupportedError(format=self)

    def open(self, transport):
        CVSProber().probe_transport(transport)
        raise NotImplementedError(self.open)