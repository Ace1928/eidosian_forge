from __future__ import annotations
import os
from enum import Enum
class ColorDepth(str, Enum):
    """
    Possible color depth values for the output.
    """
    value: str
    DEPTH_1_BIT = 'DEPTH_1_BIT'
    DEPTH_4_BIT = 'DEPTH_4_BIT'
    DEPTH_8_BIT = 'DEPTH_8_BIT'
    DEPTH_24_BIT = 'DEPTH_24_BIT'
    MONOCHROME = DEPTH_1_BIT
    ANSI_COLORS_ONLY = DEPTH_4_BIT
    DEFAULT = DEPTH_8_BIT
    TRUE_COLOR = DEPTH_24_BIT

    @classmethod
    def from_env(cls) -> ColorDepth | None:
        """
        Return the color depth if the $PROMPT_TOOLKIT_COLOR_DEPTH environment
        variable has been set.

        This is a way to enforce a certain color depth in all prompt_toolkit
        applications.
        """
        if os.environ.get('NO_COLOR'):
            return cls.DEPTH_1_BIT
        all_values = [i.value for i in ColorDepth]
        if os.environ.get('PROMPT_TOOLKIT_COLOR_DEPTH') in all_values:
            return cls(os.environ['PROMPT_TOOLKIT_COLOR_DEPTH'])
        return None

    @classmethod
    def default(cls) -> ColorDepth:
        """
        Return the default color depth for the default output.
        """
        from .defaults import create_output
        return create_output().get_default_color_depth()