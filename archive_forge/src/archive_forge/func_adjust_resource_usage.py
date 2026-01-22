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
def adjust_resource_usage(self, percentage: int) -> None:
    """Adjusts the AI model's resource usage, updating application settings accordingly."""
    log_verbose(f'Adjusting resource usage to {percentage}%.')
    self.settings_manager.update_setting('resource_usage', percentage)