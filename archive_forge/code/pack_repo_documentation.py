import re
import sys
from typing import Type
from ..lazy_import import lazy_import
import contextlib
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.index import (
from .. import errors, lockable_files, lockdir
from .. import transport as _mod_transport
from ..bzr import btree_index, index
from ..decorators import only_raises
from ..lock import LogicalLockResult
from ..repository import RepositoryWriteLockResult, _LazyListJoin
from ..trace import mutter, note, warning
from .repository import MetaDirRepository, RepositoryFormatMetaDir
from .serializer import Serializer
from .vf_repository import (MetaDirVersionedFileRepository,
Try calling the reload function, or re-raise the original exception.

        This should be called after _DirectPackAccess raises a
        RetryWithNewPacks exception. This function will handle the common logic
        of determining when the error is fatal versus being temporary.
        It will also make sure that the original exception is raised, rather
        than the RetryWithNewPacks exception.

        If this function returns, then the calling function should retry
        whatever operation was being performed. Otherwise an exception will
        be raised.

        :param retry_exc: A RetryWithNewPacks exception.
        