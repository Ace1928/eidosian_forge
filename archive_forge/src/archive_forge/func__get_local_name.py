import json
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence
import requests
def _get_local_name(self, iri: str) -> Sequence[str]:
    """
        Split IRI into prefix and local
        """
    if '#' in iri:
        tokens = iri.split('#')
        return [f'{tokens[0]}#', tokens[-1]]
    elif '/' in iri:
        tokens = iri.split('/')
        return [f'{'/'.join(tokens[0:len(tokens) - 1])}/', tokens[-1]]
    else:
        raise ValueError(f"Unexpected IRI '{iri}', contains neither '#' nor '/'.")