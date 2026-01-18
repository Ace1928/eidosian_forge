from .imports import is_tqdm_available
from ..state import PartialState
def tqdm(main_process_only: bool=True, *args, **kwargs):
    """
    Wrapper around `tqdm.tqdm` that optionally displays only on the main process.

    Args:
        main_process_only (`bool`, *optional*):
            Whether to display the progress bar only on the main process
    """
    if not is_tqdm_available():
        raise ImportError("Accelerate's `tqdm` module requires `tqdm` to be installed. Please run `pip install tqdm`.")
    disable = False
    if main_process_only:
        disable = PartialState().local_process_index != 0
    return _tqdm(*args, **kwargs, disable=disable)