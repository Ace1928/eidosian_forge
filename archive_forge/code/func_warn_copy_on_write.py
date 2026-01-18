importing `dates` and `display` ensures that keys needed by _libs
from pandas._config import config
from pandas._config import dates  # pyright: ignore[reportUnusedImport]  # noqa: F401
from pandas._config.config import (
from pandas._config.display import detect_console_encoding
def warn_copy_on_write() -> bool:
    _mode_options = _global_config['mode']
    return _mode_options['copy_on_write'] == 'warn' and _mode_options['data_manager'] == 'block'