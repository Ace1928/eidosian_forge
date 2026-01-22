import typing
from pip._vendor.tenacity import _utils
def after_nothing(retry_state: 'RetryCallState') -> None:
    """After call strategy that does nothing."""