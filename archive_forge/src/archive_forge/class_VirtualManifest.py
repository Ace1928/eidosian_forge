from __future__ import annotations
import typing as T
from typing_extensions import Literal, TypedDict, Required
class VirtualManifest(TypedDict):
    """The Representation of a virtual manifest.

    Cargo allows a root manifest that contains only a workspace, this is called
    a virtual manifest. This doesn't really map 1:1 with any meson concept,
    except perhaps the proposed "meta project".
    """
    workspace: Workspace