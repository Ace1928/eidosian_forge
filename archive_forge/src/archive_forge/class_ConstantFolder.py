import collections
from typing import Any, Callable, Dict, Optional
import torch
import torch.utils._pytree as pytree
class ConstantFolder(torch.fx.Interpreter):

    def __init__(self, gm, skip_constructors=False):
        super().__init__(gm)
        self.node_replacements: Dict[torch.fx.Node, Any] = {}
        self.replaced_uses: Dict[torch.fx.Node, int] = collections.Counter()
        self.unknown_value = object()
        self.skip_constructors: bool = skip_constructors
        self.user_to_last_uses = self.node_to_last_non_output_use()

    def is_impure(self, node: torch.fx.node.Node):
        if node.target in [torch.ops.quantized_decomposed.dequantize_per_channel.default, torch.ops.quantized_decomposed.dequantize_per_tensor.default, torch.ops.quantized_decomposed.dequantize_per_tensor.tensor]:
            return True
        return False

    def node_to_last_non_output_use(self):
        last_non_output_use = collections.defaultdict(list)
        seen_uses = set()
        output_node = next(iter(reversed(self.module.graph.nodes)))
        for node in reversed(self.module.graph.nodes):
            if node.target == 'output':
                continue

            def add_use(inp):
                if inp in seen_uses:
                    return
                seen_uses.add(inp)
                last_non_output_use[node].append(inp)
            pytree.tree_map_only(torch.fx.Node, add_use, (node.args, node.kwargs))
            if len(node.users) == 1 and output_node in node.users:
                last_non_output_use[node].append(node)
        return last_non_output_use

    def run_node(self, node):
        if node.target == 'output':

            def set_env(arg):
                self.env[arg] = self.unknown_value
            pytree.tree_map_only(torch.fx.Node, set_env, node.args)
            return super().run_node(node)
        args, kwargs = self.fetch_args_kwargs_from_env(node)
        flattened_inputs = pytree.arg_tree_leaves(*args, **kwargs)
        if self.unknown_value in flattened_inputs:
            return self.unknown_value
        if node.op == 'call_function' and node.target == aten._efficientzerotensor.default:
            return self.unknown_value
        if self.skip_constructors and node.op != 'get_attr' and (not any((isinstance(e, torch.Tensor) for e in flattened_inputs))):
            return self.unknown_value
        if isinstance(node.target, torch._ops.OpOverload) and torch.Tag.nondeterministic_seeded in node.target.tags:
            return self.unknown_value
        out = super().run_node(node)
        if node.op != 'get_attr' and isinstance(out, torch.Tensor):
            if not self.insertable_tensor_check(out):
                return out
            if self.is_impure(node):
                return self.unknown_value
            self.add_node_replacement(node, out)
            flattened_node_inps = pytree.arg_tree_leaves(*node.args, **node.kwargs)
            for n in flattened_node_inps:
                if not isinstance(n, torch.fx.Node):
                    continue
                self.replaced_uses[n] += 1
            for to_delete in self.user_to_last_uses.get(node, []):
                if self.replaced_uses[to_delete] == len(to_delete.users):
                    self.node_replacements.pop(to_delete, None)
        return out

    def insertable_tensor_check(self, tensor: torch.Tensor) -> bool:
        return True

    def add_node_replacement(self, node: torch.fx.Node, tensor: torch.Tensor) -> None:
        self.node_replacements[node] = tensor

    def run(self):
        env = {}
        for n in self.module.graph.nodes:
            if n.op == 'placeholder':
                env[n] = self.unknown_value
        return super().run(initial_env=env)