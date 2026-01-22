import warnings
from typing import Any, List, Optional
import torch
from torch import nn
from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose
class RankAllocator:
    """
    The RankAllocator for AdaLoraModel. Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.
        model: the model that we apply AdaLoRA to.

    """

    def __init__(self, model, peft_config, adapter_name):
        self.peft_config = peft_config
        self.adapter_name = adapter_name
        self.beta1 = peft_config.beta1
        self.beta2 = peft_config.beta2
        assert self.beta1 > 0 and self.beta1 < 1
        assert self.beta2 > 0 and self.beta2 < 1
        self.reset_ipt()
        self._set_budget_scheduler(model)

    def set_total_step(self, total_step):
        self.peft_config.total_step = total_step

    def reset_ipt(self):
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}

    def _set_budget_scheduler(self, model):
        self.init_bgt = 0
        self.name_set = set()
        for n, p in model.named_parameters():
            if f'lora_A.{self.adapter_name}' in n:
                self.init_bgt += p.size(0)
                self.name_set.add(n.replace('lora_A', '%s'))
        self.name_set = sorted(self.name_set)
        self.target_bgt = self.peft_config.target_r * len(self.name_set)

    def budget_schedule(self, step: int):
        tinit = self.peft_config.tinit
        tfinal = self.peft_config.tfinal
        total_step = self.peft_config.total_step
        if step <= tinit:
            budget = self.init_bgt
            mask_ind = False
        elif step > total_step - tfinal:
            budget = self.target_bgt
            mask_ind = True
        else:
            mul_coeff = 1 - (step - tinit) / (total_step - tfinal - tinit)
            budget = int((self.init_bgt - self.target_bgt) * mul_coeff ** 3 + self.target_bgt)
            mask_ind = True if step % self.peft_config.deltaT == 0 else False
        return (budget, mask_ind)

    def update_ipt(self, model):
        for n, p in model.named_parameters():
            if 'lora_' in n and self.adapter_name in n:
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.exp_avg_unc[n] = torch.zeros_like(p)
                with torch.no_grad():
                    self.ipt[n] = (p * p.grad).abs().detach()
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()

    def _element_score(self, n):
        return self.exp_avg_ipt[n] * self.exp_avg_unc[n]

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_budget(self, model, budget):
        value_ipt = {}
        vector_ipt = {}
        triplet_ipt = {}
        for n, p in model.named_parameters():
            if f'lora_A.{self.adapter_name}' in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=1, keepdim=True)
                name_m = n.replace('lora_A', '%s')
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f'lora_B.{self.adapter_name}' in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1)
                name_m = n.replace('lora_B', '%s')
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f'lora_E.{self.adapter_name}' in n:
                entry_ipt = self._element_score(n)
                name_m = n.replace('lora_E', '%s')
                value_ipt[name_m] = entry_ipt
        all_score = []
        for name_m in vector_ipt:
            ipt_E = value_ipt[name_m]
            ipt_AB = torch.cat(vector_ipt[name_m], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_m % 'lora_E'
            triplet_ipt[name_E] = sum_ipt.view(-1, 1)
            all_score.append(sum_ipt.view(-1))
        mask_threshold = torch.kthvalue(torch.cat(all_score), k=self.init_bgt - budget)[0].item()
        rank_pattern = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if f'lora_E.{self.adapter_name}' in n:
                    p.masked_fill_(triplet_ipt[n] <= mask_threshold, 0.0)
                    rank_pattern[n] = (~(triplet_ipt[n] <= mask_threshold)).view(-1).tolist()
        return rank_pattern

    def update_and_allocate(self, model, global_step, force_mask=False):
        if global_step < self.peft_config.total_step - self.peft_config.tfinal:
            self.update_ipt(model)
        budget, mask_ind = self.budget_schedule(global_step)
        if mask_ind or force_mask:
            rank_pattern = self.mask_to_budget(model, budget)
        else:
            rank_pattern = None
        return (budget, rank_pattern)

    def mask_using_rank_pattern(self, model, rank_pattern):
        is_adapter_name_truncated = False
        if self.adapter_name not in next(iter(rank_pattern.keys())):
            is_adapter_name_truncated = True
        with torch.no_grad():
            for n, p in model.named_parameters():
                if f'lora_E.{self.adapter_name}' in n:
                    key = n if not is_adapter_name_truncated else n.replace(f'.{self.adapter_name}', '')
                    mask = torch.Tensor(rank_pattern[key]).unsqueeze(-1).to(p.device)
                    p.masked_fill_(~mask.bool(), 0.0)