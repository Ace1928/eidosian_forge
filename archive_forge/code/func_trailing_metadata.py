from typing import Any, Dict, List, Optional, Tuple
import grpc
from ray.util.annotations import DeveloperAPI, PublicAPI
def trailing_metadata(self) -> List[Tuple[str, str]]:
    return self._trailing_metadata