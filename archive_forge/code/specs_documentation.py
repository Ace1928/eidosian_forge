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

        Replace the ``__deepcopy__`` member with a traits-friendly implementation.

        A bug in ``__deepcopy__`` for ``HasTraits`` results in weird cloning behaviors.
        