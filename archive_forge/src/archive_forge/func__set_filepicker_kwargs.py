import os
import re
from ...Qt import QtCore, QtGui, QtWidgets
from ..Parameter import Parameter
from .str import StrParameterItem
def _set_filepicker_kwargs(fileDlg, **kwargs):
    """Applies a dict of enum/flag kwarg opts to a file dialog"""
    NO_MATCH = object()
    for kk, vv in kwargs.items():
        formattedName = kk[0].upper() + kk[1:]
        if formattedName == 'Options':
            enumCls = fileDlg.Option
        else:
            enumCls = getattr(fileDlg, formattedName, NO_MATCH)
        setFunc = getattr(fileDlg, f'set{formattedName}', NO_MATCH)
        if enumCls is NO_MATCH or setFunc is NO_MATCH:
            continue
        if enumCls is fileDlg.Option:
            builder = fileDlg.Option(0)
            if isinstance(vv, str):
                vv = [vv]
            for flag in vv:
                curVal = getattr(enumCls, flag)
                builder |= curVal
            outEnum = enumCls(builder)
        else:
            outEnum = getattr(enumCls, vv)
        setFunc(outEnum)