import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
import math
from .. import measure, segmentation, util, color
from .._shared.version_requirements import require
Add node `n` without updating the maximum node id.

        This is a convenience method used internally.

        .. seealso:: :obj:`networkx.Graph.add_node`.