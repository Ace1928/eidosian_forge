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
class AnsibleDeprecatedChecker(BaseChecker):
    """Checks for Display.deprecated calls to ensure that the ``version``
    has not passed or met the time for removal
    """
    __implements__ = (IAstroidChecker,)
    name = 'deprecated'
    msgs = MSGS
    options = (('collection-name', {'default': None, 'type': 'string', 'metavar': '<name>', 'help': "The collection's name used to check collection names in deprecations."}), ('collection-version', {'default': None, 'type': 'string', 'metavar': '<version>', 'help': "The collection's version number used to check deprecations."}))

    def _check_date(self, node, date):
        if not isinstance(date, str):
            self.add_message('ansible-invalid-deprecated-date', node=node, args=(date,))
            return
        try:
            date_parsed = parse_isodate(date)
        except ValueError:
            self.add_message('ansible-invalid-deprecated-date', node=node, args=(date,))
            return
        if date_parsed < datetime.date.today():
            self.add_message('ansible-deprecated-date', node=node, args=(date,))

    def _check_version(self, node, version, collection_name):
        if collection_name is None:
            collection_name = 'ansible.builtin'
        if not isinstance(version, (str, float)):
            if collection_name == 'ansible.builtin':
                symbol = 'ansible-invalid-deprecated-version'
            else:
                symbol = 'collection-invalid-deprecated-version'
            self.add_message(symbol, node=node, args=(version,))
            return
        version_no = str(version)
        if collection_name == 'ansible.builtin':
            try:
                if not version_no:
                    raise ValueError('Version string should not be empty')
                loose_version = LooseVersion(str(version_no))
                if ANSIBLE_VERSION >= loose_version:
                    self.add_message('ansible-deprecated-version', node=node, args=(version,))
            except ValueError:
                self.add_message('ansible-invalid-deprecated-version', node=node, args=(version,))
        elif collection_name:
            try:
                if not version_no:
                    raise ValueError('Version string should not be empty')
                semantic_version = SemanticVersion(version_no)
                if collection_name == self.collection_name and self.collection_version is not None:
                    if self.collection_version >= semantic_version:
                        self.add_message('collection-deprecated-version', node=node, args=(version,))
                if semantic_version.major != 0 and (semantic_version.minor != 0 or semantic_version.patch != 0):
                    self.add_message('removal-version-must-be-major', node=node, args=(version,))
            except ValueError:
                self.add_message('collection-invalid-deprecated-version', node=node, args=(version,))

    @property
    def collection_name(self) -> t.Optional[str]:
        """Return the collection name, or None if ansible-core is being tested."""
        return self.linter.config.collection_name

    @property
    def collection_version(self) -> t.Optional[SemanticVersion]:
        """Return the collection version, or None if ansible-core is being tested."""
        if self.linter.config.collection_version is None:
            return None
        sem_ver = SemanticVersion(self.linter.config.collection_version)
        sem_ver.prerelease = ()
        return sem_ver

    @check_messages(*MSGS.keys())
    def visit_call(self, node):
        """Visit a call node."""
        version = None
        date = None
        collection_name = None
        try:
            funcname = _get_func_name(node)
            if funcname == 'deprecated' and 'display' in _get_expr_name(node) or funcname == 'deprecate':
                if node.keywords:
                    for keyword in node.keywords:
                        if len(node.keywords) == 1 and keyword.arg is None:
                            return
                        if keyword.arg == 'version':
                            if isinstance(keyword.value.value, astroid.Name):
                                return
                            version = keyword.value.value
                        if keyword.arg == 'date':
                            if isinstance(keyword.value.value, astroid.Name):
                                return
                            date = keyword.value.value
                        if keyword.arg == 'collection_name':
                            if isinstance(keyword.value.value, astroid.Name):
                                return
                            collection_name = keyword.value.value
                if not version and (not date):
                    try:
                        version = node.args[1].value
                    except IndexError:
                        self.add_message('ansible-deprecated-no-version', node=node)
                        return
                if version and date:
                    self.add_message('ansible-deprecated-both-version-and-date', node=node)
                if collection_name:
                    this_collection = collection_name == (self.collection_name or 'ansible.builtin')
                    if not this_collection:
                        self.add_message('wrong-collection-deprecated', node=node, args=(collection_name,))
                elif self.collection_name is not None:
                    self.add_message('ansible-deprecated-no-collection-name', node=node)
                if date:
                    self._check_date(node, date)
                elif version:
                    self._check_version(node, version, collection_name)
        except AttributeError:
            pass