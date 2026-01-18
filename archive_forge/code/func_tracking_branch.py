from git.config import GitConfigParser, SectionConstraint
from git.util import join_path
from git.exc import GitCommandError
from .symbolic import SymbolicReference
from .reference import Reference
from typing import Any, Sequence, Union, TYPE_CHECKING
from git.types import PathLike, Commit_ish
def tracking_branch(self) -> Union['RemoteReference', None]:
    """
        :return: The remote_reference we are tracking, or None if we are
            not a tracking branch."""
    from .remote import RemoteReference
    reader = self.config_reader()
    if reader.has_option(self.k_config_remote) and reader.has_option(self.k_config_remote_ref):
        ref = Head(self.repo, Head.to_full_path(strip_quotes(reader.get_value(self.k_config_remote_ref))))
        remote_refpath = RemoteReference.to_full_path(join_path(reader.get_value(self.k_config_remote), ref.name))
        return RemoteReference(self.repo, remote_refpath)
    return None