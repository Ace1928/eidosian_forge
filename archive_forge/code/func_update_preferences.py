import base64
import copy
import json
import os
import re
import shutil
import sys
import tempfile
import warnings
import zipfile
from io import BytesIO
from xml.dom import minidom
from typing_extensions import deprecated
from selenium.common.exceptions import WebDriverException
def update_preferences(self):
    """Writes the desired user prefs to disk."""
    user_prefs = os.path.join(self._profile_dir, 'user.js')
    if os.path.isfile(user_prefs):
        os.chmod(user_prefs, 420)
        self._read_existing_userjs(user_prefs)
    with open(user_prefs, 'w', encoding='utf-8') as f:
        for key, value in self._desired_preferences.items():
            f.write(f'user_pref("{key}", {json.dumps(value)});\n')