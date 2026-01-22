import breezy.bzr  # noqa: F401
from breezy import controldir
from ..commands import Command, display_command
from ..option import Option, RegistryOption
def _get_colocated_branch(self, target_controldir, name):
    from ..errors import NotBranchError
    try:
        return target_controldir.open_branch(name=name)
    except NotBranchError:
        return target_controldir.create_branch(name=name)