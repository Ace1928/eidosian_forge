from __future__ import annotations
import logging # isort:skip
import pathlib
from typing import TYPE_CHECKING, Any as any
from ..core.has_props import HasProps, abstract
from ..core.properties import (
from ..core.property.bases import Init
from ..core.property.singletons import Intrinsic
from ..core.validation import error
from ..core.validation.errors import INVALID_PROPERTY_VALUE, NOT_A_PROPERTY_OF
from ..model import Model
class CloseDialog(Callback):
    """ Close a dialog box. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    dialog = Required(Instance('.models.ui.Dialog'), help='\n    A dialog instance to close.\n\n    The behavior of this action depends on the configuration of the dialog,\n    in particular ``Dialog.close_action`` property.\n    ')