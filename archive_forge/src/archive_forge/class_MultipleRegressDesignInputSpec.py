import os
from glob import glob
from shutil import rmtree
from string import Template
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import simplify_list, ensure_list
from ...utils.misc import human_order_sorted
from ...external.due import BibTeX
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class MultipleRegressDesignInputSpec(BaseInterfaceInputSpec):
    contrasts = traits.List(traits.Either(traits.Tuple(traits.Str, traits.Enum('T'), traits.List(traits.Str), traits.List(traits.Float)), traits.Tuple(traits.Str, traits.Enum('F'), traits.List(traits.Tuple(traits.Str, traits.Enum('T'), traits.List(traits.Str), traits.List(traits.Float))))), mandatory=True, desc="List of contrasts with each contrast being a list of the form - [('name', 'stat', [condition list], [weight list])]. if session list is None or not provided, all sessions are used. For F contrasts, the condition list should contain previously defined T-contrasts without any weight list.")
    regressors = traits.Dict(traits.Str, traits.List(traits.Float), mandatory=True, desc='dictionary containing named lists of regressors')
    groups = traits.List(traits.Int, desc='list of group identifiers (defaults to single group)')