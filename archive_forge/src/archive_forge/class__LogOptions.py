from typing import Optional, Tuple, Union
from typing_extensions import TypedDict
from pytorch_lightning.utilities.exceptions import MisconfigurationException
class _LogOptions(TypedDict):
    allowed_on_step: Union[Tuple[bool], Tuple[bool, bool]]
    allowed_on_epoch: Union[Tuple[bool], Tuple[bool, bool]]
    default_on_step: bool
    default_on_epoch: bool