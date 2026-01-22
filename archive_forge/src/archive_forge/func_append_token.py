import re
from typing import Dict, Any
def append_token(self, token: Dict[str, Any]):
    """Add token to the end of token list."""
    self.tokens.append(token)