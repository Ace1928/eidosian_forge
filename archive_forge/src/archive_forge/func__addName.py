from collections import namedtuple, OrderedDict
import os
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables import otBase
from fontTools.feaLib.ast import STATNameStatement
from fontTools.otlLib.optimize.gpos import (
from fontTools.otlLib.error import OpenTypeLibError
from functools import reduce
import logging
import copy
def _addName(ttFont, value, minNameID=0, windows=True, mac=True):
    nameTable = ttFont['name']
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        names = dict(en=value)
    elif isinstance(value, dict):
        names = value
    elif isinstance(value, list):
        nameID = nameTable._findUnusedNameID()
        for nameRecord in value:
            if isinstance(nameRecord, STATNameStatement):
                nameTable.setName(nameRecord.string, nameID, nameRecord.platformID, nameRecord.platEncID, nameRecord.langID)
            else:
                raise TypeError('value must be a list of STATNameStatements')
        return nameID
    else:
        raise TypeError('value must be int, str, dict or list')
    return nameTable.addMultilingualName(names, ttFont=ttFont, windows=windows, mac=mac, minNameID=minNameID)