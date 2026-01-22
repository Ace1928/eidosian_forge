import warnings
import torch
class set_detect_anomaly:
    """Context-manager that sets the anomaly detection for the autograd engine on or off.

    ``set_detect_anomaly`` will enable or disable the autograd anomaly detection
    based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    See ``detect_anomaly`` above for details of the anomaly detection behaviour.

    Args:
        mode (bool): Flag whether to enable anomaly detection (``True``),
                     or disable (``False``).
        check_nan (bool): Flag whether to raise an error when the backward
                          generate "nan"

    """

    def __init__(self, mode: bool, check_nan: bool=True) -> None:
        self.prev = torch.is_anomaly_enabled()
        self.prev_check_nan = torch.is_anomaly_check_nan_enabled()
        torch.set_anomaly_enabled(mode, check_nan)

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: object) -> None:
        torch.set_anomaly_enabled(self.prev, self.prev_check_nan)