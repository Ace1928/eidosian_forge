from __future__ import annotations
from contextlib import contextmanager
from contextlib import nullcontext
import logging
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import ContextManager
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import Column
from sqlalchemy import literal_column
from sqlalchemy import MetaData
from sqlalchemy import PrimaryKeyConstraint
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy.engine import Engine
from sqlalchemy.engine import url as sqla_url
from sqlalchemy.engine.strategies import MockEngineStrategy
from .. import ddl
from .. import util
from ..util import sqla_compat
from ..util.compat import EncodedIO
def update_to_step(self, step: Union[RevisionStep, StampStep]) -> None:
    if step.should_delete_branch(self.heads):
        vers = step.delete_version_num
        log.debug('branch delete %s', vers)
        self._delete_version(vers)
    elif step.should_create_branch(self.heads):
        vers = step.insert_version_num
        log.debug('new branch insert %s', vers)
        self._insert_version(vers)
    elif step.should_merge_branches(self.heads):
        delete_revs, update_from_rev, update_to_rev = step.merge_branch_idents(self.heads)
        log.debug('merge, delete %s, update %s to %s', delete_revs, update_from_rev, update_to_rev)
        for delrev in delete_revs:
            self._delete_version(delrev)
        self._update_version(update_from_rev, update_to_rev)
    elif step.should_unmerge_branches(self.heads):
        update_from_rev, update_to_rev, insert_revs = step.unmerge_branch_idents(self.heads)
        log.debug('unmerge, insert %s, update %s to %s', insert_revs, update_from_rev, update_to_rev)
        for insrev in insert_revs:
            self._insert_version(insrev)
        self._update_version(update_from_rev, update_to_rev)
    else:
        from_, to_ = step.update_version_num(self.heads)
        log.debug('update %s to %s', from_, to_)
        self._update_version(from_, to_)