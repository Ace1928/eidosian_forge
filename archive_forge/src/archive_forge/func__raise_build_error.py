import lazy_loader as lazy
from skimage._shared.tester import PytestTester  # noqa
def _raise_build_error(e):
    import os.path as osp
    local_dir = osp.split(__file__)[0]
    msg = _STANDARD_MSG
    if local_dir == 'skimage':
        msg = _INPLACE_MSG
    raise ImportError(f'{e}\nIt seems that scikit-image has not been built correctly.\n{msg}')