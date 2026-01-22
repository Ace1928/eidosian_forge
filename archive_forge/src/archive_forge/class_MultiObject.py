from collections.abc import Sequence
from traits import __version__ as traits_version
import traits.api as traits
from traits.api import TraitType, Unicode
from traits.trait_base import _Undefined
from pathlib import Path
from ...utils.filemanip import path_resolve
class MultiObject(traits.List):
    """Abstract class - shared functionality of input and output MultiObject"""

    def validate(self, objekt, name, value):
        if not isinstance(value, (str, bytes)) and isinstance(value, Sequence):
            value = list(value)
        if not isdefined(value) or (isinstance(value, list) and len(value) == 0):
            return Undefined
        newvalue = value
        inner_trait = self.inner_traits()[0]
        if not isinstance(value, list) or (isinstance(inner_trait.trait_type, traits.List) and (not isinstance(inner_trait.trait_type, InputMultiObject)) and (not isinstance(value[0], list))):
            newvalue = [value]
        value = super(MultiObject, self).validate(objekt, name, newvalue)
        if value:
            return value
        self.error(objekt, name, value)