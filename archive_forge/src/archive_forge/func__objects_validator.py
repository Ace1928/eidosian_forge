from __future__ import annotations
import itertools, os, re
import typing as T
from .. import compilers
from ..build import (CustomTarget, BuildTarget,
from ..coredata import UserFeatureOption
from ..dependencies import Dependency, InternalDependency
from ..interpreterbase.decorators import KwargInfo, ContainerTypeInfo
from ..mesonlib import (File, FileMode, MachineChoice, listify, has_path_sep,
from ..programs import ExternalProgram
def _objects_validator(vals: T.List[ObjectTypes]) -> T.Optional[str]:
    non_objects: T.List[str] = []
    for val in vals:
        if isinstance(val, (str, File, ExtractedObjects)):
            continue
        else:
            non_objects.extend((o for o in val.get_outputs() if not compilers.is_object(o)))
    if non_objects:
        return f'{', '.join(non_objects)!r} are not objects'
    return None