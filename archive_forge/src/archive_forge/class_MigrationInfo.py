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
class MigrationInfo:
    """Exposes information about a migration step to a callback listener.

    The :class:`.MigrationInfo` object is available exclusively for the
    benefit of the :paramref:`.EnvironmentContext.on_version_apply`
    callback hook.

    """
    is_upgrade: bool
    'True/False: indicates whether this operation ascends or descends the\n    version tree.'
    is_stamp: bool
    'True/False: indicates whether this operation is a stamp (i.e. whether\n    it results in any actual database operations).'
    up_revision_id: Optional[str]
    'Version string corresponding to :attr:`.Revision.revision`.\n\n    In the case of a stamp operation, it is advised to use the\n    :attr:`.MigrationInfo.up_revision_ids` tuple as a stamp operation can\n    make a single movement from one or more branches down to a single\n    branchpoint, in which case there will be multiple "up" revisions.\n\n    .. seealso::\n\n        :attr:`.MigrationInfo.up_revision_ids`\n\n    '
    up_revision_ids: Tuple[str, ...]
    'Tuple of version strings corresponding to :attr:`.Revision.revision`.\n\n    In the majority of cases, this tuple will be a single value, synonymous\n    with the scalar value of :attr:`.MigrationInfo.up_revision_id`.\n    It can be multiple revision identifiers only in the case of an\n    ``alembic stamp`` operation which is moving downwards from multiple\n    branches down to their common branch point.\n\n    '
    down_revision_ids: Tuple[str, ...]
    'Tuple of strings representing the base revisions of this migration step.\n\n    If empty, this represents a root revision; otherwise, the first item\n    corresponds to :attr:`.Revision.down_revision`, and the rest are inferred\n    from dependencies.\n    '
    revision_map: RevisionMap
    'The revision map inside of which this operation occurs.'

    def __init__(self, revision_map: RevisionMap, is_upgrade: bool, is_stamp: bool, up_revisions: Union[str, Tuple[str, ...]], down_revisions: Union[str, Tuple[str, ...]]) -> None:
        self.revision_map = revision_map
        self.is_upgrade = is_upgrade
        self.is_stamp = is_stamp
        self.up_revision_ids = util.to_tuple(up_revisions, default=())
        if self.up_revision_ids:
            self.up_revision_id = self.up_revision_ids[0]
        else:
            self.up_revision_id = None
        self.down_revision_ids = util.to_tuple(down_revisions, default=())

    @property
    def is_migration(self) -> bool:
        """True/False: indicates whether this operation is a migration.

        At present this is true if and only the migration is not a stamp.
        If other operation types are added in the future, both this attribute
        and :attr:`~.MigrationInfo.is_stamp` will be false.
        """
        return not self.is_stamp

    @property
    def source_revision_ids(self) -> Tuple[str, ...]:
        """Active revisions before this migration step is applied."""
        return self.down_revision_ids if self.is_upgrade else self.up_revision_ids

    @property
    def destination_revision_ids(self) -> Tuple[str, ...]:
        """Active revisions after this migration step is applied."""
        return self.up_revision_ids if self.is_upgrade else self.down_revision_ids

    @property
    def up_revision(self) -> Optional[Revision]:
        """Get :attr:`~.MigrationInfo.up_revision_id` as
        a :class:`.Revision`.

        """
        return self.revision_map.get_revision(self.up_revision_id)

    @property
    def up_revisions(self) -> Tuple[Optional[_RevisionOrBase], ...]:
        """Get :attr:`~.MigrationInfo.up_revision_ids` as a
        :class:`.Revision`."""
        return self.revision_map.get_revisions(self.up_revision_ids)

    @property
    def down_revisions(self) -> Tuple[Optional[_RevisionOrBase], ...]:
        """Get :attr:`~.MigrationInfo.down_revision_ids` as a tuple of
        :class:`Revisions <.Revision>`."""
        return self.revision_map.get_revisions(self.down_revision_ids)

    @property
    def source_revisions(self) -> Tuple[Optional[_RevisionOrBase], ...]:
        """Get :attr:`~MigrationInfo.source_revision_ids` as a tuple of
        :class:`Revisions <.Revision>`."""
        return self.revision_map.get_revisions(self.source_revision_ids)

    @property
    def destination_revisions(self) -> Tuple[Optional[_RevisionOrBase], ...]:
        """Get :attr:`~MigrationInfo.destination_revision_ids` as a tuple of
        :class:`Revisions <.Revision>`."""
        return self.revision_map.get_revisions(self.destination_revision_ids)