from __future__ import annotations
from textwrap import dedent
from typing import (
from pandas.util._decorators import (
from pandas.core.indexers.objects import (
from pandas.core.window.doc import (
from pandas.core.window.rolling import (

        Return an indexer class that will compute the window start and end bounds

        Returns
        -------
        GroupbyIndexer
        