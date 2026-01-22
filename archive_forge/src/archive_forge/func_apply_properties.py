import tkinter as tk
from tkinter import filedialog, messagebox
import json
import logging
from typing import Callable, Dict, List, Optional, Tuple, Any
def apply_properties(self) -> None:
    """Apply the properties from the properties panel to the selected widget with detailed property application."""
    logging.info('Applying properties to the selected widget.')