from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain
import torch
import bitsandbytes.functional as F
class Optimizer8bit(torch.optim.Optimizer):

    def __init__(self, params, defaults, optim_bits=32, is_paged=False):
        """
        Base 8-bit optimizer class.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
        """
        super().__init__(params, defaults)
        self.initialized = False
        self.name2qmap = {}
        self.is_paged = is_paged
        self.page_mng = F.GlobalPageManager.get_instance()
        self.mng = GlobalOptimManager.get_instance()
        self.non_castable_tensor_keys = {'qmap1', 'qmap2', 'max1', 'max2', 'new_max1', 'new_max2', 'state1', 'state2', 'gnorm_vec', 'absmax1', 'absmax2', 'unorm_vec'}
        if optim_bits == 8:
            self.fill_qmap()

    def fill_qmap(self):
        self.name2qmap['dynamic'] = F.create_dynamic_map(signed=True)
        self.name2qmap['udynamic'] = F.create_dynamic_map(signed=False)

    def __setstate__(self, state):
        super().__setstate__(state)

    def load_state_dict(self, state_dict):
        """Load an optimizer state.

        Arguments:
            state_dict (`dict`):
                An optimizer state (should be returned from a call to `state_dict`) to load.
        """
        state_dict = deepcopy(state_dict)
        groups = self.param_groups
        saved_groups = state_dict['param_groups']
        if len(groups) != len(saved_groups):
            raise ValueError('loaded state dict has a different number of parameter groups')
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any((p_len != s_len for p_len, s_len in zip(param_lens, saved_lens))):
            raise ValueError("loaded state dict contains a parameter group that doesn't match the size of optimizer's group")
        id_map = {old_id: p for old_id, p in zip(chain.from_iterable((g['params'] for g in saved_groups)), chain.from_iterable((g['params'] for g in groups)))}

        def cast(param, value):
            """Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                if param.is_floating_point() and value.dtype != torch.uint8:
                    value = value.to(param.dtype)
                return value
            elif isinstance(value, dict):
                for k, v in value.items():
                    if k in self.non_castable_tensor_keys:
                        value[k] = v.to(param.device)
                    else:
                        value[k] = cast(param, v)
                return value
            elif isinstance(value, container_abcs.Iterable):
                return type(value)((cast(param, v) for v in value))
            else:
                return value
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def to_gpu(self):
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group['params']):
                if p in self.state:
                    values = self.state[p]
                    for k, v in values.items():
                        if isinstance(v, torch.Tensor):
                            is_paged = getattr(v, 'is_paged', False)
                            if not is_paged:
                                self.state[p][k] = v.to(p.device)

    def check_overrides(self):
        for module, attr, config in self.mng.module_weight_config_triple:
            pmodule = getattr(module, attr)
            assert pmodule is not None
            assert isinstance(pmodule, torch.Tensor) or isinstance(pmodule, torch.Parameter)
            found = False
            for gindex, group in enumerate(self.param_groups):
                if found:
                    break
                for pindex, p in enumerate(group['params']):
                    if found:
                        break
                    if id(p) == id(pmodule):
                        self.mng.pid2config[id(p)] = config
                        self.mng.index2config[gindex, pindex] = self.mng.pid2config[id(p)]
                        found = True

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Arguments:
            closure (`Callable`, *optional*, defaults to `None`):
                A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        overflows = []
        if not self.initialized:
            self.check_overrides()
            self.to_gpu()
            self.initialized = True
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    self.init_state(group, p, gindex, pindex)
                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex)
                torch.cuda.synchronize()
        if self.is_paged:
            torch.cuda.synchronize()
        return loss

    def get_config(self, gindex, pindex, group):
        config = {}
        config['betas'] = group['betas']
        config['eps'] = group['eps']
        config['weight_decay'] = group['weight_decay']
        config['lr'] = group['lr']
        config['optim_bits'] = self.args.optim_bits
        config['min_8bit_size'] = self.args.min_8bit_size
        config['percentile_clipping'] = self.args.percentile_clipping
        config['block_wise'] = self.args.block_wise
        config['max_unorm'] = self.args.max_unorm
        config['skip_zeros'] = self.args.skip_zeros
        if (gindex, pindex) in self.mng.index2config:
            config.update(self.mng.index2config[gindex, pindex])
        return config

    def init_state(self, group, p, gindex, pindex):
        raise NotImplementedError('init_state method needs to be overridden')

    def update_step(self, group, p, gindex, pindex):
        raise NotImplementedError('The update_step method needs to be overridden')

    def get_state_buffer(self, p, dtype=torch.float32):
        if not self.is_paged or p.numel() < 100000.0:
            return torch.zeros_like(p, dtype=dtype, device=p.device)
        else:
            buff = F.get_paged(*p.shape, dtype=dtype, device=p.device)
            F.fill(buff, 0)
            self.page_mng.paged_tensors.append(buff)
            return buff

    def prefetch_state(self, p):
        if self.is_paged:
            state = self.state[p]
            s1 = state['state1']
            is_paged = getattr(s1, 'is_paged', False)
            if is_paged:
                F.prefetch_tensor(state['state1'])
                if 'state2' in state:
                    F.prefetch_tensor(state['state2'])