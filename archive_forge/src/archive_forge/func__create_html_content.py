from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum
import os
import uuid
import webbrowser
import cirq_web
def _create_html_content(self, client_code: str) -> str:
    div = f'\n        <meta charset="UTF-8">\n        <div id="{self.id}"></div>\n        '
    bundle_script = self._get_bundle_script()
    return div + bundle_script + client_code