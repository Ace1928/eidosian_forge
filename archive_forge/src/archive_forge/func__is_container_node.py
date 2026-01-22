import logging
import operator
from typing import Callable, List, Optional, Set, Tuple
from functorch import make_fx
import torch
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table
def _is_container_node(node: torch.fx.Node) -> bool:
    if any((user.target == operator.getitem for user in node.users)):
        assert all((user.target == operator.getitem for user in node.users)), 'Malformed graph: a container node is used as input for non-getitem nodes.\nNode: {fmt_node}\nUsers: {fmt_users}'.format(fmt_node=node.format_node(), fmt_users='\n'.join((u.format_node() for u in node.users)))
        return True
    return False