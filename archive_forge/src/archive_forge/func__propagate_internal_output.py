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
def _propagate_internal_output(graph, node, field, connections, portinputs):
    """Propagates the given graph internal node output port
    field connections to the out-edge source node and in-edge
    destination nodes."""
    for destnode, inport, src in connections:
        if field in portinputs:
            srcnode, srcport = portinputs[field]
            if isinstance(srcport, tuple) and isinstance(src, tuple):
                src_func = srcport[1].split('\\n')[0]
                dst_func = src[1].split('\\n')[0]
                raise ValueError("Does not support two inline functions in series ('{}'  and '{}'), found when connecting {} to {}. Please use a Function node.".format(src_func, dst_func, srcnode, destnode))
            connect = graph.get_edge_data(srcnode, destnode, default={'connect': []})
            if isinstance(src, tuple):
                connect['connect'].append(((srcport, src[1], src[2]), inport))
            else:
                connect = {'connect': [(srcport, inport)]}
            old_connect = graph.get_edge_data(srcnode, destnode, default={'connect': []})
            old_connect['connect'] += connect['connect']
            graph.add_edges_from([(srcnode, destnode, old_connect)])
        else:
            value = getattr(node.inputs, field)
            if isinstance(src, tuple):
                value = evaluate_connect_function(src[1], src[2], value)
            destnode.set_input(inport, value)