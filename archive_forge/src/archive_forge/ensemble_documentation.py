import asyncio
from collections import defaultdict
from collections.abc import Hashable
from itertools import chain
from typing import (
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.load.dump import dumpd
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config, patch_config
from langchain_core.runnables.utils import (

        Perform weighted Reciprocal Rank Fusion on multiple rank lists.
        You can find more details about RRF here:
        https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

        Args:
            doc_lists: A list of rank lists, where each rank list contains unique items.

        Returns:
            list: The final aggregated list of items sorted by their weighted RRF
                    scores in descending order.
        