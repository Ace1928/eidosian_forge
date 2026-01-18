from typing import Any, List, Optional, Union
from uuid import UUID
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

        Persist a run by adding it to the traced_runs list.

        Parameters
        ----------
        run : Run
            The run to be persisted.
        