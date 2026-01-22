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
def _write_project(directory: Path, files: Project):
    """Write out the project."""
    file_list = []
    for file, contents in files:
        (directory / file).write_text(contents)
        print(f'Wrote {directory.name}{os.sep}{file}.')
        file_list.append(file)
    pyproject = {'files': file_list}
    pyproject_file = f'{directory}.pyproject'
    (directory / pyproject_file).write_text(json.dumps(pyproject))
    print(f'Wrote {directory.name}{os.sep}{pyproject_file}.')