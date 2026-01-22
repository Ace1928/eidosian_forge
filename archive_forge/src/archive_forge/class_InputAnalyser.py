import numpy as np
import tensorflow as tf
from autokeras.engine import analyser
class InputAnalyser(analyser.Analyser):

    def finalize(self):
        return