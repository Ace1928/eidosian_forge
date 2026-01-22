import configparser
from typing import Dict, List, IO, Mapping, Optional
from .default_styles import DEFAULT_STYLES
from .style import Style, StyleType
class ThemeStackError(Exception):
    """Base exception for errors related to the theme stack."""