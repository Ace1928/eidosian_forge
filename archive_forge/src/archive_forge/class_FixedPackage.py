from __future__ import annotations
import typing as T
from typing_extensions import Literal, TypedDict, Required
class FixedPackage(TypedDict, total=False):
    """A description of the Package Dictionary, fixed up."""
    name: Required[str]
    version: Required[str]
    authors: T.List[str]
    edition: EDITION
    rust_version: str
    description: str
    readme: str
    license: str
    license_file: str
    keywords: T.List[str]
    categories: T.List[str]
    workspace: str
    build: str
    links: str
    include: T.List[str]
    exclude: T.List[str]
    publish: bool
    metadata: T.Dict[str, T.Dict[str, str]]
    default_run: str
    autobins: bool
    autoexamples: bool
    autotests: bool
    autobenches: bool