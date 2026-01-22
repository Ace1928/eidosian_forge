from ... import version_info  # noqa: F401
from ... import controldir, errors
from ... import transport as _mod_transport
@staticmethod
def _has_hg_dumb_repository(transport):
    try:
        return transport.has_any(['.hg/requires', '.hg/00changelog.i'])
    except (_mod_transport.NoSuchFile, errors.PermissionDenied, errors.InvalidHttpResponse):
        return False