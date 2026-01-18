from typing import Any, Dict, Optional, Tuple
from wandb.data_types import Table
from wandb.errors import Error
def visualize(id: str, value: Table) -> Visualize:
    if not isinstance(value, Table):
        raise Error(f'Expected `value` to be `wandb.Table` type, instead got {type(value).__name__}')
    return Visualize(id=id, data=value)