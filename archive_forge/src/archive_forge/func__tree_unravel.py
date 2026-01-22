from __future__ import annotations
import itertools
import warnings
from types import FunctionType
from typing import Any, Callable
from typing_extensions import TypeAlias  # Python 3.10+
import jax.numpy as jnp
from jax import Array, lax
from jax._src import dtypes
from jax.typing import ArrayLike
from optree.ops import tree_flatten, tree_unflatten
from optree.typing import PyTreeSpec, PyTreeTypeVar
from optree.utils import safe_zip, total_order_sorted
def _tree_unravel(treespec: PyTreeSpec, unravel_flat: Callable[[Array], list[ArrayLike]], flat: Array) -> ArrayTree:
    return tree_unflatten(treespec, unravel_flat(flat))