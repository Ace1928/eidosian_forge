from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from langchain.callbacks.manager import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.schema import BaseRetriever, Document
from pydantic import Extra, Field
from mlflow.utils.annotations import experimental
Load a _RetrieverChain from a file.