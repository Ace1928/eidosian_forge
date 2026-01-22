from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
def _get_ortho(self, U, V):
    """Return B-orthonormal U with columns are B-orthogonal to V.

        .. note:: When `bparams["ortho_use_drop"] == False` then
                  `_get_ortho` is based on the Algorithm 3 from
                  [DuerschPhD2015] that is a slight modification of
                  the corresponding algorithm introduced in
                  [StathopolousWu2002]. Otherwise, the method
                  implements Algorithm 6 from [DuerschPhD2015]

        .. note:: If all U columns are B-collinear to V then the
                  returned tensor U will be empty.

        Args:

          U (Tensor) : initial approximation, size is (m, n)
          V (Tensor) : B-orthogonal external basis, size is (m, k)

        Returns:

          U (Tensor) : B-orthonormal columns (:math:`U^T B U = I`)
                       such that :math:`V^T B U=0`, size is (m, n1),
                       where `n1 = n` if `drop` is `False, otherwise
                       `n1 <= n`.
        """
    mm = torch.matmul
    mm_B = _utils.matmul
    m = self.iparams['m']
    tau_ortho = self.fparams['ortho_tol']
    tau_drop = self.fparams['ortho_tol_drop']
    tau_replace = self.fparams['ortho_tol_replace']
    i_max = self.iparams['ortho_i_max']
    j_max = self.iparams['ortho_j_max']
    use_drop = self.bparams['ortho_use_drop']
    for vkey in list(self.fvars.keys()):
        if vkey.startswith('ortho_') and vkey.endswith('_rerr'):
            self.fvars.pop(vkey)
    self.ivars.pop('ortho_i', 0)
    self.ivars.pop('ortho_j', 0)
    BV_norm = torch.norm(mm_B(self.B, V))
    BU = mm_B(self.B, U)
    VBU = mm(_utils.transpose(V), BU)
    i = j = 0
    stats = ''
    for i in range(i_max):
        U = U - mm(V, VBU)
        drop = False
        tau_svqb = tau_drop
        for j in range(j_max):
            if use_drop:
                U = self._get_svqb(U, drop, tau_svqb)
                drop = True
                tau_svqb = tau_replace
            else:
                U = self._get_svqb(U, False, tau_replace)
            if torch.numel(U) == 0:
                self.ivars['ortho_i'] = i
                self.ivars['ortho_j'] = j
                return U
            BU = mm_B(self.B, U)
            UBU = mm(_utils.transpose(U), BU)
            U_norm = torch.norm(U)
            BU_norm = torch.norm(BU)
            R = UBU - torch.eye(UBU.shape[-1], device=UBU.device, dtype=UBU.dtype)
            R_norm = torch.norm(R)
            rerr = float(R_norm) * float(BU_norm * U_norm) ** (-1)
            vkey = f'ortho_UBUmI_rerr[{i}, {j}]'
            self.fvars[vkey] = rerr
            if rerr < tau_ortho:
                break
        VBU = mm(_utils.transpose(V), BU)
        VBU_norm = torch.norm(VBU)
        U_norm = torch.norm(U)
        rerr = float(VBU_norm) * float(BV_norm * U_norm) ** (-1)
        vkey = f'ortho_VBU_rerr[{i}]'
        self.fvars[vkey] = rerr
        if rerr < tau_ortho:
            break
        if m < U.shape[-1] + V.shape[-1]:
            B = self.B
            assert B is not None
            raise ValueError(f'Overdetermined shape of U: #B-cols(={B.shape[-1]}) >= #U-cols(={U.shape[-1]}) + #V-cols(={V.shape[-1]}) must hold')
    self.ivars['ortho_i'] = i
    self.ivars['ortho_j'] = j
    return U