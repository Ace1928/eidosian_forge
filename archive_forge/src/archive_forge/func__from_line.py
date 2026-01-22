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
@classmethod
def _from_line(cls, repo: 'Repo', line: str, fetch_line: str) -> 'FetchInfo':
    """Parse information from the given line as returned by git-fetch -v
        and return a new FetchInfo object representing this information.

        We can handle a line as follows:
        "%c %-\\*s %-\\*s -> %s%s"

        Where c is either ' ', !, +, -, \\*, or =
        ! means error
        + means success forcing update
        - means a tag was updated
        * means birth of new branch or tag
        = means the head was up to date ( and not moved )
        ' ' means a fast-forward

        fetch line is the corresponding line from FETCH_HEAD, like
        acb0fa8b94ef421ad60c8507b634759a472cd56c    not-for-merge   branch '0.1.7RC' of /tmp/tmpya0vairemote_repo
        """
    match = cls._re_fetch_result.match(line)
    if match is None:
        raise ValueError('Failed to parse line: %r' % line)
    remote_local_ref_str: str
    control_character, operation, local_remote_ref, remote_local_ref_str, note = match.groups()
    control_character = cast(flagKeyLiteral, control_character)
    try:
        _new_hex_sha, _fetch_operation, fetch_note = fetch_line.split('\t')
        ref_type_name, fetch_note = fetch_note.split(' ', 1)
    except ValueError as e:
        raise ValueError('Failed to parse FETCH_HEAD line: %r' % fetch_line) from e
    flags = 0
    try:
        flags |= cls._flag_map[control_character]
    except KeyError as e:
        raise ValueError('Control character %r unknown as parsed from line %r' % (control_character, line)) from e
    old_commit: Union[Commit_ish, None] = None
    is_tag_operation = False
    if 'rejected' in operation:
        flags |= cls.REJECTED
    if 'new tag' in operation:
        flags |= cls.NEW_TAG
        is_tag_operation = True
    if 'tag update' in operation:
        flags |= cls.TAG_UPDATE
        is_tag_operation = True
    if 'new branch' in operation:
        flags |= cls.NEW_HEAD
    if '...' in operation or '..' in operation:
        split_token = '...'
        if control_character == ' ':
            split_token = split_token[:-1]
        old_commit = repo.rev_parse(operation.split(split_token)[0])
    ref_type: Optional[Type[SymbolicReference]] = None
    if remote_local_ref_str == 'FETCH_HEAD':
        ref_type = SymbolicReference
    elif ref_type_name == 'tag' or is_tag_operation:
        ref_type = TagReference
    elif ref_type_name in ('remote-tracking', 'branch'):
        ref_type = RemoteReference
    elif '/' in ref_type_name:
        ref_type = Head
    else:
        raise TypeError('Cannot handle reference type: %r' % ref_type_name)
    if ref_type is SymbolicReference:
        remote_local_ref = ref_type(repo, 'FETCH_HEAD')
    else:
        ref_path: Optional[PathLike] = None
        remote_local_ref_str = remote_local_ref_str.strip()
        if remote_local_ref_str.startswith(Reference._common_path_default + '/'):
            ref_path = remote_local_ref_str
            if ref_type is not TagReference and (not remote_local_ref_str.startswith(RemoteReference._common_path_default + '/')):
                ref_type = Reference
        elif ref_type is TagReference and 'tags/' in remote_local_ref_str:
            ref_path = join_path(RemoteReference._common_path_default, remote_local_ref_str)
        else:
            ref_path = join_path(ref_type._common_path_default, remote_local_ref_str)
        remote_local_ref = ref_type(repo, ref_path, check_path=False)
    note = note and note.strip() or ''
    return cls(remote_local_ref, flags, note, old_commit, local_remote_ref)