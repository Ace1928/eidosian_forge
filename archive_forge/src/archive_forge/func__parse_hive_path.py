import posixpath
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional
from ray.util.annotations import DeveloperAPI, PublicAPI
def _parse_hive_path(self, dir_path: str) -> Dict[str, str]:
    """Hive partition path parser.

        Returns a dictionary mapping partition keys to values given a hive-style
        partition path of the form "{key1}={value1}/{key2}={value2}/..." or an empty
        dictionary for unpartitioned files.
        """
    dirs = [d for d in dir_path.split('/') if d and d.count('=') == 1]
    kv_pairs = [d.split('=') for d in dirs] if dirs else []
    field_names = self._scheme.field_names
    if field_names and kv_pairs:
        if len(kv_pairs) != len(field_names):
            raise ValueError(f'Expected {len(field_names)} partition value(s) but found {len(kv_pairs)}: {kv_pairs}.')
        for i, field_name in enumerate(field_names):
            if kv_pairs[i][0] != field_name:
                raise ValueError(f'Expected partition key {field_name} but found {kv_pairs[i][0]}')
    return dict(kv_pairs)