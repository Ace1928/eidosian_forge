import os
from git.util import join_path
from .head import Head
from typing import Any, Iterator, NoReturn, Union, TYPE_CHECKING
from git.types import PathLike
class RemoteReference(Head):
    """A reference pointing to a remote head."""
    _common_path_default = Head._remote_common_path_default

    @classmethod
    def iter_items(cls, repo: 'Repo', common_path: Union[PathLike, None]=None, remote: Union['Remote', None]=None, *args: Any, **kwargs: Any) -> Iterator['RemoteReference']:
        """Iterate remote references, and if given, constrain them to the given remote."""
        common_path = common_path or cls._common_path_default
        if remote is not None:
            common_path = join_path(common_path, str(remote))
        return super().iter_items(repo, common_path)

    @classmethod
    def delete(cls, repo: 'Repo', *refs: 'RemoteReference', **kwargs: Any) -> None:
        """Delete the given remote references.

        :note:
            kwargs are given for comparability with the base class method as we
            should not narrow the signature.
        """
        repo.git.branch('-d', '-r', *refs)
        for ref in refs:
            try:
                os.remove(os.path.join(repo.common_dir, ref.path))
            except OSError:
                pass
            try:
                os.remove(os.path.join(repo.git_dir, ref.path))
            except OSError:
                pass

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> NoReturn:
        """Raises TypeError. Defined so the create method is disabled."""
        raise TypeError('Cannot explicitly create remote references')