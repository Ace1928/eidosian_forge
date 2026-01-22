import contextlib
import logging
import re
from git.cmd import Git, handle_process_output
from git.compat import defenc, force_text
from git.config import GitConfigParser, SectionConstraint, cp
from git.exc import GitCommandError
from git.refs import Head, Reference, RemoteReference, SymbolicReference, TagReference
from git.util import (
from typing import (
from git.types import PathLike, Literal, Commit_ish
class PushInfo(IterableObj):
    """
    Carries information about the result of a push operation of a single head::

        info = remote.push()[0]
        info.flags          # bitflags providing more information about the result
        info.local_ref      # Reference pointing to the local reference that was pushed
                            # It is None if the ref was deleted.
        info.remote_ref_string # path to the remote reference located on the remote side
        info.remote_ref # Remote Reference on the local side corresponding to
                        # the remote_ref_string. It can be a TagReference as well.
        info.old_commit # commit at which the remote_ref was standing before we pushed
                        # it to local_ref.commit. Will be None if an error was indicated
        info.summary    # summary line providing human readable english text about the push
    """
    __slots__ = ('local_ref', 'remote_ref_string', 'flags', '_old_commit_sha', '_remote', 'summary')
    _id_attribute_ = 'pushinfo'
    NEW_TAG, NEW_HEAD, NO_MATCH, REJECTED, REMOTE_REJECTED, REMOTE_FAILURE, DELETED, FORCED_UPDATE, FAST_FORWARD, UP_TO_DATE, ERROR = [1 << x for x in range(11)]
    _flag_map = {'X': NO_MATCH, '-': DELETED, '*': 0, '+': FORCED_UPDATE, ' ': FAST_FORWARD, '=': UP_TO_DATE, '!': ERROR}

    def __init__(self, flags: int, local_ref: Union[SymbolicReference, None], remote_ref_string: str, remote: 'Remote', old_commit: Optional[str]=None, summary: str='') -> None:
        """Initialize a new instance.

        local_ref: HEAD | Head | RemoteReference | TagReference | Reference | SymbolicReference | None
        """
        self.flags = flags
        self.local_ref = local_ref
        self.remote_ref_string = remote_ref_string
        self._remote = remote
        self._old_commit_sha = old_commit
        self.summary = summary

    @property
    def old_commit(self) -> Union[str, SymbolicReference, Commit_ish, None]:
        return self._old_commit_sha and self._remote.repo.commit(self._old_commit_sha) or None

    @property
    def remote_ref(self) -> Union[RemoteReference, TagReference]:
        """
        :return:
            Remote :class:`~git.refs.reference.Reference` or
            :class:`~git.refs.tag.TagReference` in the local repository corresponding to
            the :attr:`remote_ref_string` kept in this instance.
        """
        if self.remote_ref_string.startswith('refs/tags'):
            return TagReference(self._remote.repo, self.remote_ref_string)
        elif self.remote_ref_string.startswith('refs/heads'):
            remote_ref = Reference(self._remote.repo, self.remote_ref_string)
            return RemoteReference(self._remote.repo, 'refs/remotes/%s/%s' % (str(self._remote), remote_ref.name))
        else:
            raise ValueError('Could not handle remote ref: %r' % self.remote_ref_string)

    @classmethod
    def _from_line(cls, remote: 'Remote', line: str) -> 'PushInfo':
        """Create a new PushInfo instance as parsed from line which is expected to be like
        refs/heads/master:refs/heads/master 05d2687..1d0568e as bytes."""
        control_character, from_to, summary = line.split('\t', 3)
        flags = 0
        try:
            flags |= cls._flag_map[control_character]
        except KeyError as e:
            raise ValueError('Control character %r unknown as parsed from line %r' % (control_character, line)) from e
        from_ref_string, to_ref_string = from_to.split(':')
        if flags & cls.DELETED:
            from_ref: Union[SymbolicReference, None] = None
        elif from_ref_string == '(delete)':
            from_ref = None
        else:
            from_ref = Reference.from_path(remote.repo, from_ref_string)
        old_commit: Optional[str] = None
        if summary.startswith('['):
            if '[rejected]' in summary:
                flags |= cls.REJECTED
            elif '[remote rejected]' in summary:
                flags |= cls.REMOTE_REJECTED
            elif '[remote failure]' in summary:
                flags |= cls.REMOTE_FAILURE
            elif '[no match]' in summary:
                flags |= cls.ERROR
            elif '[new tag]' in summary:
                flags |= cls.NEW_TAG
            elif '[new branch]' in summary:
                flags |= cls.NEW_HEAD
        else:
            split_token = '...'
            if control_character == ' ':
                split_token = '..'
            old_sha, _new_sha = summary.split(' ')[0].split(split_token)
            old_commit = old_sha
        return PushInfo(flags, from_ref, to_ref_string, remote, old_commit, summary)

    @classmethod
    def iter_items(cls, repo: 'Repo', *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError