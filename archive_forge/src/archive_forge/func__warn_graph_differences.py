import inspect
import math
import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torchvision
from torch import fx, nn
from torch.fx.graph_module import _copy_attr
def _warn_graph_differences(train_tracer: NodePathTracer, eval_tracer: NodePathTracer):
    """
    Utility function for warning the user if there are differences between
    the train graph nodes and the eval graph nodes.
    """
    train_nodes = list(train_tracer.node_to_qualname.values())
    eval_nodes = list(eval_tracer.node_to_qualname.values())
    if len(train_nodes) == len(eval_nodes) and all((t == e for t, e in zip(train_nodes, eval_nodes))):
        return
    suggestion_msg = 'When choosing nodes for feature extraction, you may need to specify output nodes for train and eval mode separately.'
    if _is_subseq(train_nodes, eval_nodes):
        msg = 'NOTE: The nodes obtained by tracing the model in eval mode are a subsequence of those obtained in train mode. '
    elif _is_subseq(eval_nodes, train_nodes):
        msg = 'NOTE: The nodes obtained by tracing the model in train mode are a subsequence of those obtained in eval mode. '
    else:
        msg = 'The nodes obtained by tracing the model in train mode are different to those obtained in eval mode. '
    warnings.warn(msg + suggestion_msg)