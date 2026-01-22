from abc import ABCMeta, abstractmethod
import enum
from typing import Optional, Union, Tuple
from scipy.sparse import spmatrix
import numpy as np
from pyomo.contrib.pynumero.sparse.base_block import BaseBlockVector, BaseBlockMatrix
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
class LinearSolverInterface(object, metaclass=ABCMeta):

    @abstractmethod
    def solve(self, matrix: Union[spmatrix, BlockMatrix], rhs: Union[np.ndarray, BlockVector], raise_on_error: bool=True) -> Tuple[Optional[Union[np.ndarray, BlockVector]], LinearSolverResults]:
        pass