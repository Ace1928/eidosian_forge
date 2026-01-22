import base64
import re
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
from langchain_core.runnables.graph import (
def _escape_node_label(node_label: str) -> str:
    """Escapes the node label for Mermaid syntax."""
    return re.sub('[^a-zA-Z-_]', '_', node_label)