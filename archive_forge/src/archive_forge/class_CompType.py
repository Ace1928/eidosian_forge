from __future__ import annotations
import argparse
import enum
import os
import typing as t
class CompType(enum.Enum):
    """
    Bash COMP_TYPE argument completion types.
    For documentation, see: https://www.gnu.org/software/bash/manual/html_node/Bash-Variables.html#index-COMP_005fTYPE
    """
    COMPLETION = '\t'
    '\n    Standard completion, typically triggered by a single tab.\n    '
    MENU_COMPLETION = '%'
    '\n    Menu completion, which cycles through each completion instead of showing a list.\n    For help using this feature, see: https://stackoverflow.com/questions/12044574/getting-complete-and-menu-complete-to-work-together\n    '
    LIST = '?'
    '\n    Standard list, typically triggered by a double tab.\n    '
    LIST_AMBIGUOUS = '!'
    '\n    Listing with `show-all-if-ambiguous` set.\n    For documentation, see https://www.gnu.org/software/bash/manual/html_node/Readline-Init-File-Syntax.html#index-show_002dall_002dif_002dambiguous\n    For additional details, see: https://unix.stackexchange.com/questions/614123/explanation-of-bash-completion-comp-type\n    '
    LIST_UNMODIFIED = '@'
    '\n    Listing with `show-all-if-unmodified` set.\n    For documentation, see https://www.gnu.org/software/bash/manual/html_node/Readline-Init-File-Syntax.html#index-show_002dall_002dif_002dunmodified\n    For additional details, see: : https://unix.stackexchange.com/questions/614123/explanation-of-bash-completion-comp-type\n    '

    @property
    def list_mode(self) -> bool:
        """True if completion is running in list mode, otherwise False."""
        return self in (CompType.LIST, CompType.LIST_AMBIGUOUS, CompType.LIST_UNMODIFIED)