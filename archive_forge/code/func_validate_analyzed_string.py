import pathlib
from xml.etree.ElementTree import Element
from typing import Optional
from ..exceptions import ElementPathRuntimeError
def validate_analyzed_string(root: Element) -> None:
    raise ElementPathRuntimeError('not schema-aware')