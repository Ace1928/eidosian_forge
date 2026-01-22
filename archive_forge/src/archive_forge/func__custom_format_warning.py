import warnings
from pathlib import Path
from typing import Optional, Type, Union
from lightning_fabric.utilities.rank_zero import LightningDeprecationWarning
def _custom_format_warning(message: Union[Warning, str], category: Type[Warning], filename: str, lineno: int, line: Optional[str]=None) -> str:
    """Custom formatting that avoids an extra line in case warnings are emitted from the `rank_zero`-functions."""
    if _is_path_in_lightning(Path(filename)):
        return f'{filename}:{lineno}: {message}\n'
    return _default_format_warning(message, category, filename, lineno, line)