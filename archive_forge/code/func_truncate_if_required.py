from typing import List
from typing import Optional
from _pytest.assertion import util
from _pytest.config import Config
from _pytest.nodes import Item
def truncate_if_required(explanation: List[str], item: Item, max_length: Optional[int]=None) -> List[str]:
    """Truncate this assertion explanation if the given test item is eligible."""
    if _should_truncate_item(item):
        return _truncate_explanation(explanation)
    return explanation