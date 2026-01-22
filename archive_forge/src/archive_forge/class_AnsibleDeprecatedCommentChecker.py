from __future__ import annotations
import datetime
import functools
import json
import re
import shlex
import typing as t
from tokenize import COMMENT, TokenInfo
import astroid
from pylint.checkers import BaseChecker, BaseTokenChecker
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.six import string_types
from ansible.release import __version__ as ansible_version_raw
from ansible.utils.version import SemanticVersion
class AnsibleDeprecatedCommentChecker(BaseTokenChecker):
    """Checks for ``# deprecated:`` comments to ensure that the ``version``
    has not passed or met the time for removal
    """
    __implements__ = (ITokenChecker,)
    name = 'deprecated-comment'
    msgs = {'E9601': ('Deprecated core version (%r) found: %s', 'ansible-deprecated-version-comment', "Used when a '# deprecated:' comment specifies a version less than or equal to the current version of Ansible", {'minversion': (2, 6)}), 'E9602': ('Deprecated comment contains invalid keys %r', 'ansible-deprecated-version-comment-invalid-key', "Used when a '#deprecated:' comment specifies invalid data", {'minversion': (2, 6)}), 'E9603': ('Deprecated comment missing version', 'ansible-deprecated-version-comment-missing-version', "Used when a '#deprecated:' comment specifies invalid data", {'minversion': (2, 6)}), 'E9604': ('Deprecated python version (%r) found: %s', 'ansible-deprecated-python-version-comment', "Used when a '#deprecated:' comment specifies a python version less than or equal to the minimum python version", {'minversion': (2, 6)}), 'E9605': ('Deprecated comment contains invalid version %r: %s', 'ansible-deprecated-version-comment-invalid-version', "Used when a '#deprecated:' comment specifies an invalid version", {'minversion': (2, 6)})}
    options = (('min-python-version-db', {'default': None, 'type': 'string', 'metavar': '<path>', 'help': 'The path to the DB mapping paths to minimum Python versions.'}),)

    def process_tokens(self, tokens: list[TokenInfo]) -> None:
        for token in tokens:
            if token.type == COMMENT:
                self._process_comment(token)

    def _deprecated_string_to_dict(self, token: TokenInfo, string: str) -> dict[str, str]:
        valid_keys = {'description', 'core_version', 'python_version'}
        data = dict.fromkeys(valid_keys)
        for opt in shlex.split(string):
            if '=' not in opt:
                data[opt] = None
                continue
            key, _sep, value = opt.partition('=')
            data[key] = value
        if not any((data['core_version'], data['python_version'])):
            self.add_message('ansible-deprecated-version-comment-missing-version', line=token.start[0], col_offset=token.start[1])
        bad = set(data).difference(valid_keys)
        if bad:
            self.add_message('ansible-deprecated-version-comment-invalid-key', line=token.start[0], col_offset=token.start[1], args=(','.join(bad),))
        return data

    @functools.cached_property
    def _min_python_version_db(self) -> dict[str, str]:
        """A dictionary of absolute file paths and their minimum required Python version."""
        with open(self.linter.config.min_python_version_db) as db_file:
            return json.load(db_file)

    def _process_python_version(self, token: TokenInfo, data: dict[str, str]) -> None:
        current_file = self.linter.current_file
        check_version = self._min_python_version_db[current_file]
        try:
            if LooseVersion(data['python_version']) < LooseVersion(check_version):
                self.add_message('ansible-deprecated-python-version-comment', line=token.start[0], col_offset=token.start[1], args=(data['python_version'], data['description'] or 'description not provided'))
        except (ValueError, TypeError) as exc:
            self.add_message('ansible-deprecated-version-comment-invalid-version', line=token.start[0], col_offset=token.start[1], args=(data['python_version'], exc))

    def _process_core_version(self, token: TokenInfo, data: dict[str, str]) -> None:
        try:
            if ANSIBLE_VERSION >= LooseVersion(data['core_version']):
                self.add_message('ansible-deprecated-version-comment', line=token.start[0], col_offset=token.start[1], args=(data['core_version'], data['description'] or 'description not provided'))
        except (ValueError, TypeError) as exc:
            self.add_message('ansible-deprecated-version-comment-invalid-version', line=token.start[0], col_offset=token.start[1], args=(data['core_version'], exc))

    def _process_comment(self, token: TokenInfo) -> None:
        if token.string.startswith('# deprecated:'):
            data = self._deprecated_string_to_dict(token, token.string[13:].strip())
            if data['core_version']:
                self._process_core_version(token, data)
            if data['python_version']:
                self._process_python_version(token, data)