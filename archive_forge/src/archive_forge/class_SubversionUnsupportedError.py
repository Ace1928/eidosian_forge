from ... import version_info  # noqa: F401
from ... import controldir, errors
from ... import transport as _mod_transport
from ...revisionspec import revspec_registry
class SubversionUnsupportedError(errors.UnsupportedVcs):
    vcs = 'svn'
    _fmt = 'Subversion branches are not yet supported. To interoperate with Subversion branches, use fastimport.'