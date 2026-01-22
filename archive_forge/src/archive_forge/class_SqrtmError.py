import numpy as np
from scipy._lib._util import _asarray_validated
from ._misc import norm
from .lapack import ztrsyl, dtrsyl
from ._decomp_schur import schur, rsf2csf
from ._matfuncs_sqrtm_triu import within_block_loop  # noqa: E402
class SqrtmError(np.linalg.LinAlgError):
    pass