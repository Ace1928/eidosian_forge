import email.feedparser
import email.header
import email.message
import email.parser
import email.policy
import sys
import typing
from typing import Dict, List, Optional, Tuple, Union, cast
def _parse_project_urls(data: List[str]) -> Dict[str, str]:
    """Parse a list of label/URL string pairings separated by a comma."""
    urls = {}
    for pair in data:
        parts = [p.strip() for p in pair.split(',', 1)]
        parts.extend([''] * max(0, 2 - len(parts)))
        label, url = parts
        if label in urls:
            raise KeyError('duplicate labels in project urls')
        urls[label] = url
    return urls