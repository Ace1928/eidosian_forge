import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import List, Tuple
from PySide6.QtWidgets import QApplication, QMainWindow
import QtQuick.Controls
from pathlib import Path
from PySide6.QtGui import QGuiApplication
from PySide6.QtCore import QUrl
from PySide6.QtQml import QQmlApplicationEngine
def _qml_project() -> Project:
    """Create a QML project."""
    return [('main.py', _QUICK_MAIN), ('main.qml', _QUICK_FORM)]