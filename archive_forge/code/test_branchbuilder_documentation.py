from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
It's possible (albeit awkward) to move an existing dir to the root
        in a single snapshot by using unversion then flush then add.
        