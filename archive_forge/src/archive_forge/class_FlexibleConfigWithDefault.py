import collections
import pickle
import pytest
import networkx as nx
from networkx.utils.configs import Config
class FlexibleConfigWithDefault(Config, strict=False):
    x: int = 0