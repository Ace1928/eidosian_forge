from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
class SampleExtraBranchFormat(_mod_branch.BranchFormat):
    """A sample format that is not usable in a metadir."""

    def get_format_string(self):
        return None

    def network_name(self):
        return 'extra'

    def initialize(self, a_controldir, name=None):
        raise NotImplementedError(self.initialize)

    def open(self, transport, name=None, _found=False, ignore_fallbacks=False, possible_transports=None):
        raise NotImplementedError(self.open)