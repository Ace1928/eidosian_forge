import os
from inspect import isclass
from copy import deepcopy
from warnings import warn
from packaging.version import Version
from traits.trait_errors import TraitError
from traits.trait_handlers import TraitDictObject, TraitListObject
from ...utils.filemanip import md5, hash_infile, hash_timestamp
from .traits_extension import (
from ... import config, __version__
class TraitedSpec(BaseTraitedSpec):
    """Create a subclass with strict traits.

    This is used in 90% of the cases.
    """
    _ = traits.Disallow