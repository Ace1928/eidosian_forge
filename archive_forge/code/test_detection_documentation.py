import logging
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.dependencies import networkx_available, matplotlib_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from pyomo.contrib.community_detection.detection import (
from pyomo.contrib.community_detection.community_graph import generate_model_graph
from pyomo.solvers.tests.models.LP_unbounded import LP_unbounded
from pyomo.solvers.tests.models.QP_simple import QP_simple
from pyomo.solvers.tests.models.LP_inactive_index import LP_inactive_index
from pyomo.solvers.tests.models.SOS1_simple import SOS1_simple

    This is the testing helper function, which collects all 24 possible community maps for a model and
    all 24 possible partitions of the respective model graphs (based on Louvain community detection done on
    the networkX graph of the model) - (24 is the total combinations possible assuming that we have
    provided a seed value for the community detection).

    This function generates all combinations of the parameters for detect_communities by looping through the types
    of community maps we can have and then looping through every combination of the other parameters (as seen below).
    The inner for loop is used to create 8 different combinations of parameter values for with_objective,
    weighted_graph, and use_only_active_components by counting from 0 to 7 and then interpreting this number as a
    binary value, which will then be used to assign a True/False value to each of the three parameters.

    Parameters
    ----------
    model: Block
        a Pyomo model or block to be used for community detection

    Returns
    -------
    list_of_community_maps:
        a list of 24 CommunityMap objects (the 24 community maps correspond to all the
        combinations of parameters that are possible for detect_communities).
        The parameters for creating the 24 community maps are:
        type_of_community_map = ['constraint', 'variable', 'bipartite']
        with_objective = [False, True]
        weighted_graph = [False, True]
        use_only_active_components = [None, True]

    list_of_partitions:
        a list of 24 partitions based on Louvain community detection done on the networkX graph of the model (the
        24 partitions correspond to all the combinations of parameters that are possible for detect_communities).
        The parameters for creating the 24 partitions are:
        type_of_community_map = ['constraint', 'variable', 'bipartite']
        with_objective = [False, True]
        weighted_graph = [False, True]
        use_only_active_components = [None, True]
    