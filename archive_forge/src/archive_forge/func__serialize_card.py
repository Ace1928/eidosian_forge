import abc
import json
import logging
import os
import time
import traceback
from enum import Enum
from typing import Any, Dict, List, Optional
import yaml
from mlflow.recipes.cards import CARD_HTML_NAME, CARD_PICKLE_NAME, BaseCard, FailureCard
from mlflow.recipes.utils import get_recipe_name
from mlflow.recipes.utils.step import display_html
from mlflow.tracking import MlflowClient
from mlflow.utils.databricks_utils import is_in_databricks_runtime
def _serialize_card(self, start_timestamp: float, output_directory: str) -> None:
    if self.step_card is None:
        return
    execution_duration = time.time() - start_timestamp
    tab = self.step_card.get_tab('Run Summary')
    if tab is not None:
        tab.add_markdown('EXE_DURATION', f'**Run duration (s)**: {execution_duration:.3g}')
        tab.add_markdown('LAST_UPDATE_TIME', f'**Last updated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}')
    self.step_card.save(path=output_directory)
    self.step_card.save_as_html(path=output_directory)