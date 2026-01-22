import sys
from typing import Any
import pytorch_lightning as pl
class IPUPrecisionPlugin:

    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError('The `IPUPrecisionPlugin` class has been moved to an external package. Install the extension package as `pip install lightning-graphcore` and import with `from lightning_graphcore import IPUPrecisionPlugin`. Please see: https://github.com/Lightning-AI/lightning-Graphcore for more details.')