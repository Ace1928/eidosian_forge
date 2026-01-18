from typing import Type
from ..lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import transport as _mod_transport
from ..repository import InterRepository, IsInWriteGroupError, Repository
from .repository import RepositoryFormatMetaDir
from .serializer import Serializer
from .vf_repository import (InterSameDataRepository,
See InterRepository.search_missing_revision_ids().