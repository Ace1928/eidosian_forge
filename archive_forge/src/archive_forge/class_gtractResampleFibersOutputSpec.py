import os
from ...base import (
class gtractResampleFibersOutputSpec(TraitedSpec):
    outputTract = File(desc='Required: name of output vtkPolydata file containing tract lines and the point data collected along them.', exists=True)