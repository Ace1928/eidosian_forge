import numpy as np
import skimage.graph.mcp as mcp
from skimage._shared.testing import assert_array_equal
Simple MCP subclass that allows the front to travel
    a certain distance from the seed point, and uses a constant
    cost factor that is independent of the cost array.
    