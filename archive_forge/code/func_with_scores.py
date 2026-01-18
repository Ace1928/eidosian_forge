from typing import List, Optional, Union
def with_scores(self) -> 'Query':
    """Ask the engine to return document search scores."""
    self._with_scores = True
    return self