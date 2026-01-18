import collections
import enum
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables.otConverters import (
from fontTools.misc.roundTools import otRound

    Helps to populate things derived from BaseTable from maps, tuples, etc.

    A table of lifecycle callbacks may be provided to add logic beyond what is possible
    based on otData info for the target class. See BuildCallbacks.
    