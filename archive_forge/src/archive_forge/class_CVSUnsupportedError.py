from ... import version_info  # noqa: F401
from ... import controldir, errors
from ...transport import register_transport_proto
class CVSUnsupportedError(errors.UnsupportedVcs):
    vcs = 'cvs'
    _fmt = 'CVS working trees are not supported. To convert CVS projects to bzr, please see http://bazaar-vcs.org/BzrMigration and/or https://launchpad.net/launchpad-bazaar/+faq/26.'