import numpy as np
from scipy.optimize import minimize
import GPy
from GPy.kern import Kern
from GPy.core import Param
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
Select the number of datapoints to keep, using cross validation