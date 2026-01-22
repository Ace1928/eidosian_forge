import shutil
import tempfile
from ... import errors
from ... import merge as _mod_merge
from ... import trace
from ...i18n import gettext
from ...mutabletree import MutableTree
from ...revisiontree import RevisionTree
from .quilt import QuiltPatches
class NoUnapplyingMerger(_mod_merge.Merge3Merger):
    _no_quilt_unapplying = True