from fontTools.misc import psCharStrings
from fontTools import ttLib
from fontTools.pens.basePen import NullPen
from fontTools.misc.roundTools import otRound
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.subset.util import _add_method, _uniq_sort
def _delete_empty_subrs(private_dict):
    if hasattr(private_dict, 'Subrs') and (not private_dict.Subrs):
        if 'Subrs' in private_dict.rawDict:
            del private_dict.rawDict['Subrs']
        del private_dict.Subrs