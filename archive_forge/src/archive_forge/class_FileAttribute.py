import dataclasses
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
class FileAttribute(GitlabAttribute):

    @staticmethod
    def get_file_name(attr_name: Optional[str]=None) -> Optional[str]:
        return attr_name