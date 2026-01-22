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
def add_markdown(self, name: str, markdown: str) -> CardTab:
    """
        Adds markdown to the card replacing the variable name in the CardTab template.

        Args:
            name: Name of the variable in the CardTab Jinja2 template.
            markdown: The markdown content.

        Returns:
            The updated card tab instance.
        """
    from markdown import markdown as md_to_html
    self.add_html(name, md_to_html(markdown))
    return self