import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from wandb.sdk.lib.runid import generate_id
Convert a list of dictionaries to a pair of column names and corresponding values, with the option to order specific dictionaries.

        :param args: The arguments passed to the API client.
        :param kwargs: The keyword arguments passed to the API client.
        :param parsed_response: The parsed response from the API.
        :param start_time: The start time of the API request.
        :param time_elapsed: The time elapsed during the API request.
        :param response_type: The type of the API response.
        :param table_column_order: The desired order of columns in the resulting table.
        :param default_model: The default model to use if not specified in the response.
        :return: A dictionary containing the formatted response.
        