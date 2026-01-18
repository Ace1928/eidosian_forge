import os
from .. import check, osutils
from ..commit import PointlessCommit
from . import TestCaseWithTransport
from .features import SymlinkFeature
from .matchers import RevisionHistoryMatches
Commit merge of two trees with no overlapping files.