import os
import sys
import pickle
from collections import defaultdict
import re
from copy import deepcopy
from glob import glob
from pathlib import Path
from traceback import format_exception
from hashlib import sha1
from functools import reduce
import numpy as np
from ... import logging, config
from ...utils.filemanip import (
from ...utils.misc import str2bool
from ...utils.functions import create_function_from_source
from ...interfaces.base.traits_extension import (
from ...interfaces.base.support import Bunch, InterfaceResult
from ...interfaces.base import CommandLine
from ...interfaces.utility import IdentityInterface
from ...utils.provenance import ProvStore, pm, nipype_ns, get_id
from inspect import signature
def _node_ports(graph, node):
    """Return the given node's input and output ports

    The return value is the (inputs, outputs) dictionaries.
    The inputs is a {destination field: (source node, source field)}
    dictionary.
    The outputs is a {source field: destination items} dictionary,
    where each destination item is a
    (destination node, destination field, source field) tuple.
    """
    portinputs = {}
    portoutputs = {}
    for u, _, d in graph.in_edges(node, data=True):
        for src, dest in d['connect']:
            portinputs[dest] = (u, src)
    for _, v, d in graph.out_edges(node, data=True):
        for src, dest in d['connect']:
            if isinstance(src, tuple):
                srcport = src[0]
            else:
                srcport = src
            if srcport not in portoutputs:
                portoutputs[srcport] = []
            portoutputs[srcport].append((v, dest, src))
    return (portinputs, portoutputs)