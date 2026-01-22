from __future__ import annotations
import base64
import html
import logging
import os
import pathlib
import pickle
import random
import re
import string
from io import StringIO
from typing import Optional, Union
from packaging.version import Version
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
def add_html(self, name: str, html_content: str) -> CardTab:
    """
        Adds html to the CardTab.

        Args:
            name: Name of the variable in the Jinja2 template.
            html_content: The html to replace the named template variable.

        Returns:
            The updated card instance.
        """
    if name not in self._variables:
        raise MlflowException(f"{name} is not a valid template variable defined in template: '{self.template}'", error_code=INVALID_PARAMETER_VALUE)
    self._context[name] = html_content
    return self