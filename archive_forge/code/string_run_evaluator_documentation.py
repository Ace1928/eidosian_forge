from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.load.dump import dumpd
from langchain_core.load.load import load
from langchain_core.load.serializable import Serializable
from langchain_core.messages import BaseMessage, get_buffer_string, messages_from_dict
from langsmith import EvaluationResult, RunEvaluator
from langsmith.schemas import DataType, Example, Run
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import StringEvaluator
from langchain.schema import RUN_KEY

        Create a StringRunEvaluatorChain from an evaluator and the run and dataset types.

        This method provides an easy way to instantiate a StringRunEvaluatorChain, by
        taking an evaluator and information about the type of run and the data.
        The method supports LLM and chain runs.

        Args:
            evaluator (StringEvaluator): The string evaluator to use.
            run_type (str): The type of run being evaluated.
                Supported types are LLM and Chain.
            data_type (DataType): The type of dataset used in the run.
            input_key (str, optional): The key used to map the input from the run.
            prediction_key (str, optional): The key used to map the prediction from the run.
            reference_key (str, optional): The key used to map the reference from the dataset.
            tags (List[str], optional): List of tags to attach to the evaluation chain.

        Returns:
            StringRunEvaluatorChain: The instantiated evaluation chain.

        Raises:
            ValueError: If the run type is not supported, or if the evaluator requires a
                reference from the dataset but the reference key is not provided.

        