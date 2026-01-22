import os
from ...base import (
class BRAINSConstellationModelerOutputSpec(TraitedSpec):
    outputModel = File(desc=',               The full filename of the output model file.,             ', exists=True)
    resultsDir = Directory(desc=',               The directory for the results to be written.,             ', exists=True)