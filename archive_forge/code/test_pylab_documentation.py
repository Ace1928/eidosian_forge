import itertools
import os
import warnings
import pytest
import networkx as nx
Users should get a warning when they specify a non-default value for
    one of the kwargs that applies only to edges drawn with FancyArrowPatches,
    but FancyArrowPatches aren't being used under the hood.