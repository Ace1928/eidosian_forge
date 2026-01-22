import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class CreateJacobianDeterminantImageOutputSpec(TraitedSpec):
    jacobian_image = File(exists=True, desc='jacobian image')