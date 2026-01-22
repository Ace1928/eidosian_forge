from ..base import TraitedSpec, traits, File
from .base import AFNICommand, AFNICommandInputSpec, AFNICommandOutputSpec
class SVMTrainOutputSpec(TraitedSpec):
    out_file = File(desc='sum of weighted linear support vectors file name')
    model = File(desc='brik containing the SVM model file name')
    alphas = File(desc='output alphas file name')