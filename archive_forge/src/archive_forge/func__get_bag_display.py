from abc import abstractmethod
from typing import Any, List, Optional
from ..dataset import Dataset, DatasetDisplay, get_dataset_display
@get_dataset_display.candidate(lambda ds: isinstance(ds, Bag), priority=0.1)
def _get_bag_display(ds: Bag):
    return BagDisplay(ds)