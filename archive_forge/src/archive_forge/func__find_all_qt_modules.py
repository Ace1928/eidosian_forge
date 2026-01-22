import os
import sys
from pathlib import Path
from textwrap import dedent
def _find_all_qt_modules():
    location = Path(__file__).resolve().parent
    in_build = Path('/home/qt/work/pyside/pyside-setup/build/qfpa-py3.11-qt6.6.2-64bit-release/build/pyside6') in location.parents
    if in_build:
        return __all__
    files = os.listdir(location)
    unordered = set((name[:-4] for name in files if name.startswith('Qt') and name.endswith('.pyi')))
    ordered_part = __all__
    result = []
    for name in ordered_part:
        if name in unordered:
            result.append(name)
            unordered.remove(name)
    result.extend(unordered)
    return result