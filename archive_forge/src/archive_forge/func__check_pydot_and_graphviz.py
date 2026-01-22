from ray.dag import DAGNode
import os
import tempfile
from ray.dag.utils import _DAGNodeNameGenerator
from ray.util.annotations import DeveloperAPI
def _check_pydot_and_graphviz():
    """Check if pydot and graphviz are installed.

    pydot and graphviz are required for plotting. We check this
    during runtime rather than adding them to Ray dependencies.

    """
    try:
        import pydot
    except ImportError:
        raise ImportError('pydot is required to plot DAG, install it with `pip install pydot`.')
    try:
        pydot.Dot.create(pydot.Dot())
    except (OSError, pydot.InvocationException):
        raise ImportError('graphviz is required to plot DAG, download it from https://graphviz.gitlab.io/download/')