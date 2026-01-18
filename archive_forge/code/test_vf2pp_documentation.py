import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism

                G1                           G2
        x--------------x              x--------------x
        | \            |              | \            |
        |  x-------x   |              |  x-------x   |
        |  |       |   |              |  |       |   |
        |  x-------x   |              |  x-------x   |
        | /            |              |            \ |
        x--------------x              x--------------x
        