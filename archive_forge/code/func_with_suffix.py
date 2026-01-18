from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type, Union
from wandb import util
@classmethod
def with_suffix(cls: Type['WBValue'], name: str, filetype: str='json') -> str:
    """Get the name with the appropriate suffix.

        Args:
            name (str): the name of the file
            filetype (str, optional): the filetype to use. Defaults to "json".

        Returns:
            str: a filename which is suffixed with it's `_log_type` followed by the
                filetype.
        """
    if cls._log_type is not None:
        suffix = cls._log_type + '.' + filetype
    else:
        suffix = filetype
    if not name.endswith(suffix):
        return name + '.' + suffix
    return name