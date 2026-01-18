importing `dates` and `display` ensures that keys needed by _libs
from pandas._config import config
from pandas._config import dates  # pyright: ignore[reportUnusedImport]  # noqa: F401
from pandas._config.config import (
from pandas._config.display import detect_console_encoding
def using_nullable_dtypes() -> bool:
    _mode_options = _global_config['mode']
    return _mode_options['nullable_dtypes']