from __future__ import annotations
from .. import mesonlib, mlog
from .disabler import Disabler
from .exceptions import InterpreterException, InvalidArguments
from ._unholder import _unholder
from dataclasses import dataclass
from functools import wraps
import abc
import itertools
import copy
import typing as T
class FeatureNew(FeatureCheckBase):
    """Checks for new features"""
    feature_registry = {}

    @staticmethod
    def check_version(target_version: str, feature_version: str) -> bool:
        return mesonlib.version_compare_condition_with_min(target_version, feature_version)

    @staticmethod
    def get_warning_str_prefix(tv: str) -> str:
        return f"Project specifies a minimum meson_version '{tv}' but uses features which were added in newer versions:"

    @staticmethod
    def get_notice_str_prefix(tv: str) -> str:
        return ''

    def log_usage_warning(self, tv: str, location: T.Optional['mparser.BaseNode']) -> None:
        args = ['Project targets', f"'{tv}'", 'but uses feature introduced in', f"'{self.feature_version}':", f'{self.feature_name}.']
        if self.extra_message:
            args.append(self.extra_message)
        mlog.warning(*args, location=location)