from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast
import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList
class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. _Gshard: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate: gate network
        expert: expert network
        group: group to use for all-to-all communication
    """

    def __init__(self, gate: Module, experts: Union[Module, ModuleList], group: Optional[Any]=None) -> None:
        super().__init__()
        self.gate = gate
        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])
        self.group = group if group is not None else dist.group.WORLD
        for expert in self.experts:
            for p in experts.parameters():
                p.expert = True
        self.world_size = dist.get_world_size(self.group)
        self.num_local_experts = len(self.experts)

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        assert len(input) == 1, 'only single input Tensor supported'
        assert len(input[0].shape) == 3, 'input Tensor must have dimensions: (s)equence, (t)oken, (m)odel'
        assert input[0].shape[0] % len(self.experts) == 0, 'num tokens must be order of number of local experts'
        d_model = input[0].shape[2]
        reshaped_input = input[0].reshape(-1, d_model)
        self.l_aux, combine_weights, dispatch_mask = self.gate(reshaped_input)
        dispatched_input = torch.einsum('sec,sm->ecm', dispatch_mask.float(), reshaped_input)
        dispatched_input = _AllToAll.apply(self.group, dispatched_input)
        dispatched_input = dispatched_input.reshape(self.world_size, self.num_local_experts, -1, d_model)
        chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            expert_outputs += [expert(chunk)]
        expert_output = torch.cat(expert_outputs, dim=1)
        expert_output = _AllToAll.apply(self.group, expert_output)
        expert_output = expert_output.reshape(self.world_size * self.num_local_experts, -1, d_model)
        combined_output = torch.einsum('sec,ecm->sm', combine_weights, expert_output)
        return combined_output.reshape(input[0].shape)