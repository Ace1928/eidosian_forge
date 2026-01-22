from .base import FSLCommand, FSLCommandInputSpec
from ..base import TraitedSpec, File, traits
class B0CalcOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='filename of B0 output volume')