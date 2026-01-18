import math
import pytest
import networkx as nx
from networkx.algorithms.planar_drawing import triangulate_embedding
Checks if pos conforms to the planar embedding

    Returns true iff the neighbors are actually oriented in the orientation
    specified of the embedding
    