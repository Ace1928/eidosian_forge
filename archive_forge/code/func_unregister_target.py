from __future__ import annotations
import contextlib
import logging
import typing as t
import uuid
from traitlets.utils.importstring import import_item
import comm
def unregister_target(self, target_name: str, f: CommTargetCallback) -> CommTargetCallback:
    """Unregister a callable registered with register_target"""
    return self.targets.pop(target_name)