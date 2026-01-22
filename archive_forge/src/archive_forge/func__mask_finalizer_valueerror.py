from __future__ import annotations
import sys
import typing as t
def _mask_finalizer_valueerror(ur: t.Any) -> None:
    """Mask only ValueErrors from finalizing abandoned generators; delegate everything else"""
    if ur.exc_type is ValueError and 'generator already executing' in str(ur.exc_value):
        return
    sys.__unraisablehook__(ur)