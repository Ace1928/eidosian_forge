from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
Gets the objective coefficient for the quadratic term associated to the product between two variables.

        The ordering of the input variables does not matter.

        Args:
          first_variable_id: The first variable in the product.
          second_variable_id: The second variable in the product.

        Returns:
          The value of the coefficient.
        