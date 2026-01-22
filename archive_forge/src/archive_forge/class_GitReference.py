import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, Union
from wandb.sdk.launch.errors import LaunchError
@dataclass
class GitReference:

    def __init__(self, remote: str, ref: Optional[str]=None) -> None:
        """Initialize a reference from a remote and ref.

        Arguments:
            remote: A remote URL or URI.
            ref: A branch, tag, or commit hash.
        """
        self.uri = remote
        self.ref = ref

    @property
    def url(self) -> Optional[str]:
        return self.uri

    def fetch(self, dst_dir: str) -> None:
        """Fetch the repo into dst_dir and refine githubref based on what we learn."""
        import git
        repo = git.Repo.init(dst_dir)
        self.path = repo.working_dir
        origin = repo.create_remote('origin', self.uri or '')
        try:
            origin.fetch()
        except git.exc.GitCommandError as e:
            raise LaunchError(f'Unable to fetch from git remote repository {self.url}:\n{e}')
        ref: Union[git.RemoteReference, str]
        if self.ref:
            if self.ref in origin.refs:
                ref = origin.refs[self.ref]
            else:
                ref = self.ref
            head = repo.create_head(self.ref, ref)
            head.checkout()
            self.commit_hash = head.commit.hexsha
        else:
            default_branch = None
            for ref in repo.references:
                if hasattr(ref, 'tag'):
                    continue
                refname = ref.name
                if refname.startswith('origin/'):
                    refname = refname[7:]
                if refname == 'main':
                    default_branch = 'main'
                    break
                if refname == 'master':
                    default_branch = 'master'
            if not default_branch:
                raise LaunchError(f'Unable to determine branch or commit to checkout from {self.url}')
            self.default_branch = default_branch
            self.ref = default_branch
            head = repo.create_head(default_branch, origin.refs[default_branch])
            head.checkout()
            self.commit_hash = head.commit.hexsha
        repo.submodule_update(init=True, recursive=True)