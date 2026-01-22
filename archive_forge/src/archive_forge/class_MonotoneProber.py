from ... import version_info  # noqa: F401
from ... import controldir, errors
class MonotoneProber(controldir.Prober):

    @classmethod
    def priority(klass, transport):
        return 100

    @classmethod
    def probe_transport(klass, transport):
        """Our format is present if the transport has a '_MTN/' subdir."""
        if transport.has('_MTN'):
            return MonotoneDirFormat()
        raise errors.NotBranchError(path=transport.base)

    @classmethod
    def known_formats(cls):
        return [MonotoneDirFormat()]