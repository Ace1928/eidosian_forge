import os
from copy import deepcopy
import pytest
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces import utility as niu
from .... import config
from ..utils import (
class StrPathConfuserOutputSpec(nib.TraitedSpec):
    out_tuple = nib.traits.Tuple(nib.File, nib.traits.String)
    out_dict_path = nib.traits.Dict(nib.traits.String, nib.File(exists=True))
    out_dict_str = nib.traits.DictStrStr()
    out_list = nib.traits.List(nib.traits.String)
    out_str = nib.traits.String()
    out_path = nib.File(exists=True)