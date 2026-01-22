from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class AnnotationDictionaryAttributes:
    """TABLE 8.15 Entries common to all annotation dictionaries."""
    Type = '/Type'
    Subtype = '/Subtype'
    Rect = '/Rect'
    Contents = '/Contents'
    P = '/P'
    NM = '/NM'
    M = '/M'
    F = '/F'
    AP = '/AP'
    AS = '/AS'
    DA = '/DA'
    Border = '/Border'
    C = '/C'
    StructParent = '/StructParent'
    OC = '/OC'