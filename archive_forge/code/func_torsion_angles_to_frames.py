from typing import Dict, Tuple, overload
import torch
import torch.types
from torch import nn
from . import residue_constants as rc
from .rigid_utils import Rigid, Rotation
from .tensor_utils import batched_gather
def torsion_angles_to_frames(r: Rigid, alpha: torch.Tensor, aatype: torch.Tensor, rrgdf: torch.Tensor) -> Rigid:
    default_4x4 = rrgdf[aatype, ...]
    default_r = r.from_tensor_4x4(default_4x4)
    bb_rot = alpha.new_zeros((*(1,) * len(alpha.shape[:-1]), 2))
    bb_rot[..., 1] = 1
    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2)
    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha
    all_frames = default_r.compose(Rigid(Rotation(rot_mats=all_rots), None))
    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]
    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)
    all_frames_to_bb = Rigid.cat([all_frames[..., :5], chi2_frame_to_bb.unsqueeze(-1), chi3_frame_to_bb.unsqueeze(-1), chi4_frame_to_bb.unsqueeze(-1)], dim=-1)
    all_frames_to_global = r[..., None].compose(all_frames_to_bb)
    return all_frames_to_global