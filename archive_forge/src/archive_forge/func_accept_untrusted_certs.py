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
@accept_untrusted_certs.setter
@deprecated('Allowing untrusted certs is toggled in the Options class')
def accept_untrusted_certs(self, value) -> None:
    if not isinstance(value, bool):
        raise WebDriverException('Please pass in a Boolean to this call')
    self.set_preference('webdriver_accept_untrusted_certs', value)