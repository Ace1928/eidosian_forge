from collections import OrderedDict
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble._base import _set_random_states
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
class RevParamPipeline(Pipeline):

    def get_params(self, *args, **kwargs):
        params = Pipeline.get_params(self, *args, **kwargs).items()
        return OrderedDict(sorted(params, reverse=True))