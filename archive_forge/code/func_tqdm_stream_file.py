import io
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union
from tqdm.auto import tqdm as old_tqdm
from ..constants import HF_HUB_DISABLE_PROGRESS_BARS
@contextmanager
def tqdm_stream_file(path: Union[Path, str]) -> Iterator[io.BufferedReader]:
    """
    Open a file as binary and wrap the `read` method to display a progress bar when it's streamed.

    First implemented in `transformers` in 2019 but removed when switched to git-lfs. Used in `huggingface_hub` to show
    progress bar when uploading an LFS file to the Hub. See github.com/huggingface/transformers/pull/2078#discussion_r354739608
    for implementation details.

    Note: currently implementation handles only files stored on disk as it is the most common use case. Could be
          extended to stream any `BinaryIO` object but we might have to debug some corner cases.

    Example:
    ```py
    >>> with tqdm_stream_file("config.json") as f:
    >>>     requests.put(url, data=f)
    config.json: 100%|█████████████████████████| 8.19k/8.19k [00:02<00:00, 3.72kB/s]
    ```
    """
    if isinstance(path, str):
        path = Path(path)
    with path.open('rb') as f:
        total_size = path.stat().st_size
        pbar = tqdm(unit='B', unit_scale=True, total=total_size, initial=0, desc=path.name)
        f_read = f.read

        def _inner_read(size: Optional[int]=-1) -> bytes:
            data = f_read(size)
            pbar.update(len(data))
            return data
        f.read = _inner_read
        yield f
        pbar.close()