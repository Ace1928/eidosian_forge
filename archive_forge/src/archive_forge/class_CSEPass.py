from typing import Dict, Tuple, Any
import torch
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.utils._pytree import tree_flatten
from torch.fx import GraphModule, Graph
from torch.fx import Node
@torch.fx._compatibility.compatibility(is_backward_compatible=False)
class CSEPass(PassBase):

    def __init__(self, banned_ops=None):
        """
        This version of CSE Pass aims to be dialect agnostic, and it's implemented purely based on the connectivity between fx.Node.

        For functional dialects, user would only need to specify the random ops in ban list.

        Warning: CSE Pass cannot be safely applied on a FX graph in non-functional dialects.
        If your dialect contains stateful operators, please customized the banned_ops.

        """
        if banned_ops is None:
            banned_ops = set()
        self.banned_ops = banned_ops
        super().__init__()

    def call(self, graph_module: GraphModule) -> PassResult:
        """
        Return a new copy of torch.fx.GraphModule with CSE applied to the input graph

        Example usage:

        from torch.fx.experimental.proxy_tensor import make_fx
        def f(a):
            b = a * a
            c = a * a
            return b+c

        p = CSEPass()
        traced_graph = make_fx(f)(torch.tensor(1))
        print(traced_graph)
        result = p(traced_graph)
        print(result.graph_module)
        """

        def get_aten_target(node):
            if hasattr(node.target, 'overloadpacket'):
                return node.target.overloadpacket
            return node.target
        modified = False
        new_graph = Graph()
        env: Dict[Node, Node] = {}
        hash_env: Dict[Tuple[torch._ops.OpOverload, int], Node] = {}
        token_map: Dict[Tuple[torch._ops.OpOverload, int], Dict[str, Any]] = {}
        for n in graph_module.graph.nodes:
            if n.op == 'placeholder' or n.op == 'output' or n.op == 'get_attr' or (get_aten_target(n) in self.banned_ops):
                new_node = new_graph.node_copy(n, lambda x: env[x])
                env[n] = new_node
            else:

                def substitute(arg_list):
                    arg_list, spec = tree_flatten(arg_list)
                    for i in range(len(arg_list)):
                        v = arg_list[i]
                        if isinstance(v, Node) and v in env:
                            arg_list[i] = env[v]
                    return (tuple(arg_list), spec)
                args, args_spec = substitute(n.args)
                kwargs, kwargs_spec = substitute(n.kwargs)
                token = {'target': n.target, 'args': args, 'args_spec': args_spec, 'kwargs': kwargs, 'kwargs_spec': kwargs_spec}
                hash_arg = hash((args, kwargs))
                hash_val = (n.target, hash_arg)
                hash_val_in_hash_env = hash_val in hash_env
                if hash_val_in_hash_env and token_map[hash_val] == token:
                    modified = True
                    env[n] = hash_env[hash_val]
                    continue
                new_node = new_graph.node_copy(n, lambda x: env[x])
                env[n] = new_node
                if not hash_val_in_hash_env:
                    hash_env[hash_val] = new_node
                    token_map[hash_val] = token
        csed_gm = GraphModule(graph_module, new_graph)
        return PassResult(csed_gm, modified)