import collections
import pickle
import pytest
import networkx as nx
from networkx.utils.configs import Config
class FlexibleConfig(Config, strict=False):
    x: int