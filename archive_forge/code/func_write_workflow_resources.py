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
def write_workflow_resources(graph, filename=None, append=None):
    """
    Generate a JSON file with profiling traces that can be loaded
    in a pandas DataFrame or processed with JavaScript like D3.js
    """
    import simplejson as json
    filename = config.get('monitoring', 'summary_file', filename)
    if not filename:
        filename = os.path.join(os.getcwd(), 'resource_monitor.json')
    if append is None:
        append = str2bool(config.get('monitoring', 'summary_append', 'true'))
    big_dict = {'time': [], 'name': [], 'interface': [], 'rss_GiB': [], 'vms_GiB': [], 'cpus': [], 'mapnode': [], 'params': []}
    if append and os.path.isfile(filename):
        with open(filename, 'r') as rsf:
            big_dict = json.load(rsf)
    for _, node in enumerate(graph.nodes()):
        nodename = node.fullname
        classname = node.interface.__class__.__name__
        params = ''
        if node.parameterization:
            params = '_'.join(['{}'.format(p) for p in node.parameterization])
        try:
            rt_list = node.result.runtime
        except Exception:
            logger.warning('Could not access runtime info for node %s (%s interface)', nodename, classname)
            continue
        if not isinstance(rt_list, list):
            rt_list = [rt_list]
        for subidx, runtime in enumerate(rt_list):
            try:
                nsamples = len(runtime.prof_dict['time'])
            except AttributeError:
                logger.warning('Could not retrieve profiling information for node "%s" (mapflow %d/%d).', nodename, subidx + 1, len(rt_list))
                continue
            for key in ['time', 'cpus', 'rss_GiB', 'vms_GiB']:
                big_dict[key] += runtime.prof_dict[key]
            big_dict['interface'] += [classname] * nsamples
            big_dict['name'] += [nodename] * nsamples
            big_dict['mapnode'] += [subidx] * nsamples
            big_dict['params'] += [params] * nsamples
    with open(filename, 'w') as rsf:
        json.dump(big_dict, rsf, ensure_ascii=False)
    return filename