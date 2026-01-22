from typing import Type
from lightning_utilities.core.enums import StrEnum
from typing_extensions import Literal
class MDMCAverageMethod(EnumStr):
    """Enum to represent multi-dim multi-class average method."""

    @staticmethod
    def _name() -> str:
        return 'MDMC Average method'
    GLOBAL = 'global'
    SAMPLEWISE = 'samplewise'