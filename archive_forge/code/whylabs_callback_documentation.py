from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.utils import get_from_env
Instantiate whylogs Logger from params.

        Args:
            api_key (Optional[str]): WhyLabs API key. Optional because the preferred
                way to specify the API key is with environment variable
                WHYLABS_API_KEY.
            org_id (Optional[str]): WhyLabs organization id to write profiles to.
                If not set must be specified in environment variable
                WHYLABS_DEFAULT_ORG_ID.
            dataset_id (Optional[str]): The model or dataset this callback is gathering
                telemetry for. If not set must be specified in environment variable
                WHYLABS_DEFAULT_DATASET_ID.
            sentiment (bool): If True will initialize a model to perform
                sentiment analysis compound score. Defaults to False and will not gather
                this metric.
            toxicity (bool): If True will initialize a model to score
                toxicity. Defaults to False and will not gather this metric.
            themes (bool): If True will initialize a model to calculate
                distance to configured themes. Defaults to None and will not gather this
                metric.
            logger (Optional[Logger]): If specified will bind the configured logger as
                the telemetry gathering agent. Defaults to LangKit schema with periodic
                WhyLabs writer.
        