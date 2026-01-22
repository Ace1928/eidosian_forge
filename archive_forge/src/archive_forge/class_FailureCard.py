from __future__ import annotations
import base64
import html
import logging
import os
import pathlib
import pickle
import random
import re
import string
from io import StringIO
from typing import Optional, Union
from packaging.version import Version
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
class FailureCard(BaseCard):
    """
    Step card providing information about a failed step execution, including a stacktrace.

    TODO: Migrate the failure card to a tab-based card, removing this class and its associated
          HTML template in the process.
    """

    def __init__(self, recipe_name: str, step_name: str, failure_traceback: str, output_directory: str):
        super().__init__(recipe_name=recipe_name, step_name=step_name)
        self.add_tab('Step Status', '{{ STEP_STATUS }}').add_html('STEP_STATUS', '<p><strong>Step status: <span style="color:red">Failed</span></strong></p>')
        self.add_tab('Stacktrace', "<div class='stacktrace-container'><p style='margin-top:0px'><code>{{ STACKTRACE|e }}</code></p></div>").add_html('STACKTRACE', str(failure_traceback))
        warning_output_path = os.path.join(output_directory, 'warning_logs.txt')
        if os.path.exists(warning_output_path):
            with open(warning_output_path) as f:
                self.add_tab('Warning Logs', '{{ STEP_WARNINGS }}').add_html('STEP_WARNINGS', f'<pre>{f.read()}</pre>')