import sys
import torch
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget,
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os
import json
import logging
from typing import Dict, Tuple, Optional, Any
from datetime import datetime
from huggingface_hub import HfApi
def update_setting(self, key: str, value: Any) -> None:
    """Updates a given setting and ensures persistence of changes."""
    self.settings[key] = value
    self.persist_settings()