from pyomo.contrib.pynumero.interfaces.nlp import NLP
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from pyomo.contrib.pynumero.linalg.ma27_interface import MA27
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import numpy as np
from scipy.sparse import tril
import pyomo.environ as pe
from pyomo import dae
from pyomo.common.timing import TicTocTimer
import time
from pyomo.contrib.pynumero.linalg.base import LinearSolverInterface, LinearSolverStatus

    An example of a simple SQP algorithm for
    equality-constrained NLPs.

    Parameters
    ----------
    nlp: NLP
        A PyNumero NLP
    max_iter: int
        The maximum number of iterations
    tol: float
        The convergence tolerance
    