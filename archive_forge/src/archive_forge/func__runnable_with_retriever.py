import contextlib
import inspect
import logging
import uuid
import warnings
from copy import deepcopy
from packaging.version import Version
import mlflow
from mlflow.entities import RunTag
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.autologging_utils import (
from mlflow.utils.autologging_utils.safety import _resolve_extra_tags
def _runnable_with_retriever(model):
    from langchain.schema import BaseRetriever
    with contextlib.suppress(ImportError):
        from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnableSequence
        from langchain.schema.runnable.passthrough import RunnableAssign
        if isinstance(model, RunnableBranch):
            return any((_runnable_with_retriever(runnable) for _, runnable in model.branches))
        if isinstance(model, RunnableParallel):
            return any((_runnable_with_retriever(runnable) for runnable in model.steps.values()))
        if isinstance(model, RunnableSequence):
            return any((_runnable_with_retriever(runnable) for runnable in model.steps))
        if isinstance(model, RunnableAssign):
            return _runnable_with_retriever(model.mapper)
    return isinstance(model, BaseRetriever)