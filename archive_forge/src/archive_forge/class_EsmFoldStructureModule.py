import math
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from ...integrations.deepspeed import is_deepspeed_available
from ...modeling_outputs import ModelOutput
from ...utils import (
from .configuration_esm import EsmConfig
from .modeling_esm import ESM_START_DOCSTRING, EsmModel, EsmPreTrainedModel
from .openfold_utils import (
class EsmFoldStructureModule(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_norm_s = LayerNorm(config.sequence_dim)
        self.layer_norm_z = LayerNorm(config.pairwise_dim)
        self.linear_in = EsmFoldLinear(config.sequence_dim, config.sequence_dim)
        self.ipa = EsmFoldInvariantPointAttention(config)
        self.ipa_dropout = nn.Dropout(config.dropout_rate)
        self.layer_norm_ipa = LayerNorm(config.sequence_dim)
        self.transition = EsmFoldStructureModuleTransition(config)
        self.bb_update = EsmFoldBackboneUpdate(config)
        self.angle_resnet = EsmFoldAngleResnet(config)

    def forward(self, evoformer_output_dict, aatype, mask=None, _offload_inference=False):
        """
        Args:
            evoformer_output_dict:
                Dictionary containing:
                    "single":
                        [*, N_res, C_s] single representation
                    "pair":
                        [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        s = evoformer_output_dict['single']
        if mask is None:
            mask = s.new_ones(s.shape[:-1])
        s = self.layer_norm_s(s)
        z = self.layer_norm_z(evoformer_output_dict['pair'])
        z_reference_list = None
        if _offload_inference:
            assert sys.getrefcount(evoformer_output_dict['pair']) == 2
            evoformer_output_dict['pair'] = evoformer_output_dict['pair'].cpu()
            z_reference_list = [z]
            z = None
        s_initial = s
        s = self.linear_in(s)
        rigids = Rigid.identity(s.shape[:-1], s.dtype, s.device, self.training, fmt='quat')
        outputs = []
        for i in range(self.config.num_blocks):
            s = s + self.ipa(s, z, rigids, mask, _offload_inference=_offload_inference, _z_reference_list=z_reference_list)
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)
            rigids = rigids.compose_q_update_vec(self.bb_update(s))
            backb_to_global = Rigid(Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None), rigids.get_trans())
            backb_to_global = backb_to_global.scale_translation(self.config.trans_scale_factor)
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)
            all_frames_to_global = self.torsion_angles_to_frames(backb_to_global, angles, aatype)
            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(all_frames_to_global, aatype)
            scaled_rigids = rigids.scale_translation(self.config.trans_scale_factor)
            preds = {'frames': scaled_rigids.to_tensor_7(), 'sidechain_frames': all_frames_to_global.to_tensor_4x4(), 'unnormalized_angles': unnormalized_angles, 'angles': angles, 'positions': pred_xyz, 'states': s}
            outputs.append(preds)
            rigids = rigids.stop_rot_gradient()
        del z, z_reference_list
        if _offload_inference:
            evoformer_output_dict['pair'] = evoformer_output_dict['pair'].to(s.device)
        outputs = dict_multimap(torch.stack, outputs)
        outputs['single'] = s
        return outputs

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, 'default_frames'):
            self.register_buffer('default_frames', torch.tensor(residue_constants.restype_rigid_group_default_frame, dtype=float_dtype, device=device, requires_grad=False), persistent=False)
        if not hasattr(self, 'group_idx'):
            self.register_buffer('group_idx', torch.tensor(residue_constants.restype_atom14_to_rigid_group, device=device, requires_grad=False), persistent=False)
        if not hasattr(self, 'atom_mask'):
            self.register_buffer('atom_mask', torch.tensor(residue_constants.restype_atom14_mask, dtype=float_dtype, device=device, requires_grad=False), persistent=False)
        if not hasattr(self, 'lit_positions'):
            self.register_buffer('lit_positions', torch.tensor(residue_constants.restype_atom14_rigid_group_positions, dtype=float_dtype, device=device, requires_grad=False), persistent=False)

    def torsion_angles_to_frames(self, r, alpha, f):
        self._init_residue_constants(alpha.dtype, alpha.device)
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(self, r, f):
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(r, f, self.default_frames, self.group_idx, self.atom_mask, self.lit_positions)