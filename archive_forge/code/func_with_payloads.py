from typing import List, Optional, Union
def with_payloads(self) -> 'Query':
    """Ask the engine to return document payloads."""
    self._with_payloads = True
    return self