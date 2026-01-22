from typing import Any, Callable, Optional
import wandb
class PreInitObject:

    def __init__(self, name: str, destination: Optional[Any]=None) -> None:
        self._name = name
        if destination is not None:
            self.__doc__ = destination.__doc__

    def __getitem__(self, key: str) -> None:
        raise wandb.Error(f'You must call wandb.init() before {self._name}[{key!r}]')

    def __setitem__(self, key: str, value: Any) -> Any:
        raise wandb.Error(f'You must call wandb.init() before {self._name}[{key!r}]')

    def __setattr__(self, key: str, value: Any) -> Any:
        if not key.startswith('_'):
            raise wandb.Error(f'You must call wandb.init() before {self._name}.{key}')
        else:
            return object.__setattr__(self, key, value)

    def __getattr__(self, key: str) -> Any:
        if not key.startswith('_'):
            raise wandb.Error(f'You must call wandb.init() before {self._name}.{key}')
        else:
            raise AttributeError