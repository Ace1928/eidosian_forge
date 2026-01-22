import railroad
from pip._vendor import pyparsing
import typing
from typing import (
from jinja2 import Template
from io import StringIO
import inspect
class AnnotatedItem(railroad.Group):
    """
    Simple subclass of Group that creates an annotation label
    """

    def __init__(self, label: str, item):
        super().__init__(item=item, label='[{}]'.format(label) if label else label)