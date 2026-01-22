import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
def _get_labels(self) -> Tuple[List[str], List[str]]:
    """Get node and edge labels from the Neptune statistics summary"""
    summary = self._get_summary()
    n_labels = summary['nodeLabels']
    e_labels = summary['edgeLabels']
    return (n_labels, e_labels)