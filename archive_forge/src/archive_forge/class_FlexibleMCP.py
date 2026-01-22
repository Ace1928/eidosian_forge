import numpy as np
import skimage.graph.mcp as mcp
from skimage._shared.testing import assert_array_equal
class FlexibleMCP(mcp.MCP_Flexible):
    """Simple MCP subclass that allows the front to travel
    a certain distance from the seed point, and uses a constant
    cost factor that is independent of the cost array.
    """

    def _reset(self):
        mcp.MCP_Flexible._reset(self)
        self._distance = np.zeros((8, 8), dtype=np.float32).ravel()

    def goal_reached(self, index, cumcost):
        if self._distance[index] > 4:
            return 2
        else:
            return 0

    def travel_cost(self, index, new_index, offset_length):
        return 1.0

    def examine_neighbor(self, index, new_index, offset_length):
        pass

    def update_node(self, index, new_index, offset_length):
        self._distance[new_index] = self._distance[index] + 1