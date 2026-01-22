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
def add_tab(self, name, html_template) -> CardTab:
    """
        Add a new tab with arbitrary content.

        Args:
            name: A string representing the name of the tab.
            html_template: A string representing the HTML template for the card content.
        """
    tab = CardTab(name, html_template)
    self._tabs.append((name, tab))
    return tab