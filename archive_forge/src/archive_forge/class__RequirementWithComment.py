import re
from distutils.version import LooseVersion
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Union
from pkg_resources import Requirement, yield_lines
class _RequirementWithComment(Requirement):
    strict_string = '# strict'

    def __init__(self, *args: Any, comment: str='', pip_argument: Optional[str]=None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.comment = comment
        if not (pip_argument is None or pip_argument):
            raise RuntimeError(f'wrong pip argument: {pip_argument}')
        self.pip_argument = pip_argument
        self.strict = self.strict_string in comment.lower()

    def adjust(self, unfreeze: str) -> str:
        """Remove version restrictions unless they are strict.

        >>> _RequirementWithComment("arrow<=1.2.2,>=1.2.0", comment="# anything").adjust("none")
        'arrow<=1.2.2,>=1.2.0'
        >>> _RequirementWithComment("arrow<=1.2.2,>=1.2.0", comment="# strict").adjust("none")
        'arrow<=1.2.2,>=1.2.0  # strict'
        >>> _RequirementWithComment("arrow<=1.2.2,>=1.2.0", comment="# my name").adjust("all")
        'arrow>=1.2.0'
        >>> _RequirementWithComment("arrow>=1.2.0, <=1.2.2", comment="# strict").adjust("all")
        'arrow<=1.2.2,>=1.2.0  # strict'
        >>> _RequirementWithComment("arrow").adjust("all")
        'arrow'
        >>> _RequirementWithComment("arrow>=1.2.0, <=1.2.2", comment="# cool").adjust("major")
        'arrow<2.0,>=1.2.0'
        >>> _RequirementWithComment("arrow>=1.2.0, <=1.2.2", comment="# strict").adjust("major")
        'arrow<=1.2.2,>=1.2.0  # strict'
        >>> _RequirementWithComment("arrow>=1.2.0").adjust("major")
        'arrow>=1.2.0'
        >>> _RequirementWithComment("arrow").adjust("major")
        'arrow'

        """
        out = str(self)
        if self.strict:
            return f'{out}  {self.strict_string}'
        if unfreeze == 'major':
            for operator, version in self.specs:
                if operator in ('<', '<='):
                    major = LooseVersion(version).version[0]
                    return out.replace(f'{operator}{version}', f'<{int(major) + 1}.0')
        elif unfreeze == 'all':
            for operator, version in self.specs:
                if operator in ('<', '<='):
                    return out.replace(f'{operator}{version},', '')
        elif unfreeze != 'none':
            raise ValueError(f'Unexpected unfreeze: {unfreeze!r} value.')
        return out