from __future__ import annotations
import typing as ty
import warnings
from .deprecator import Deprecator
from .pkg_info import cmp_pkg_version
class ModuleProxy:
    """Proxy for module that may not yet have been imported

    Parameters
    ----------
    module_name : str
        Full module name e.g. ``nibabel.minc``

    Examples
    --------

    ::
        arr = np.arange(24).reshape((2, 3, 4))
        nifti1 = ModuleProxy('nibabel.nifti1')
        nifti1_image = nifti1.Nifti1Image(arr, np.eye(4))

    So, the ``nifti1`` object is a proxy that will import the required module
    when you do attribute access and return the attributes of the imported
    module.
    """

    def __init__(self, module_name: str) -> None:
        self._module_name = module_name

    def __getattr__(self, key: str) -> ty.Any:
        mod = __import__(self._module_name, fromlist=[''])
        return getattr(mod, key)

    def __repr__(self) -> str:
        return f'<module proxy for {self._module_name}>'