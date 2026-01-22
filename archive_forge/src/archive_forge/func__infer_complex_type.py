import ast  # Importing the abstract syntax tree module to manipulate and analyze Python abstract syntax grammatically. Documentation: https://docs.python.org/3/library/ast.html
import json  # Importing the JSON module to encode and decode JSON data. Documentation: https://docs.python.org/3/library/json.html
import tkinter as tk  # Importing the tkinter module for creating graphical user interfaces. Documentation: https://docs.python.org/3/library/tkinter.html
from tkinter import (
import os  # Importing the os module to interact with the operating system. Documentation: https://docs.python.org/3/library/os.html
import logging  # Importing the logging module to enable logging capabilities. Documentation: https://docs.python.org/3/library/logging.html
from typing import (
import docstring_parser  # Importing the docstring_parser module to parse Python docstrings. Documentation: https://pypi.org/project/docstring-parser/
from concurrent.futures import (
def _infer_complex_type(self, annotation: ast.AST) -> str:
    """
        Handles complex type annotations including nested generics and Python 3.10+ features like TypeGuard.

        Parameters:
            annotation (ast.AST): The AST node representing a type annotation.

        Returns:
            str: A string representation of the inferred type.

        This method uses recursive calls to handle nested annotations and logs any failures in type inference.
        """
    try:
        if isinstance(annotation, ast.Subscript):
            base: str = self._infer_complex_type(annotation.value)
            if hasattr(annotation.slice, 'value'):
                index: str = self._infer_complex_type(annotation.slice.value)
            else:
                index: str = ', '.join((self._infer_complex_type(s.value) for s in annotation.slice.values))
            return f'{base}[{index}]'
        elif isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Tuple):
            elements: str = ', '.join((self._infer_complex_type(el) for el in annotation.elts))
            return f'Tuple[{elements}]'
        elif isinstance(annotation, ast.Attribute):
            return f'{self._infer_complex_type(annotation.value)}.{annotation.attr}'
        elif isinstance(annotation, ast.Constant):
            if annotation.value == 'TypeGuard':
                return 'TypeGuard'
            return str(annotation.value)
        return 'Unknown'
    except Exception as e:
        logging.warning(f'Failed to infer type: {e}')
        return 'Unknown'