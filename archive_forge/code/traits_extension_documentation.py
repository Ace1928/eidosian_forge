from collections.abc import Sequence
from traits import __version__ as traits_version
import traits.api as traits
from traits.api import TraitType, Unicode
from traits.trait_base import _Undefined
from pathlib import Path
from ...utils.filemanip import path_resolve
Create an ImageFile trait.