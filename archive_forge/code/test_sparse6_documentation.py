import tempfile
from io import BytesIO
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
Unit tests for writing graphs in the sparse6 format.

    Most of the test cases were checked against the sparse6 encoder in Sage.

    