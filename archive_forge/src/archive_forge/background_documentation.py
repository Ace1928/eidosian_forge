from typing import Any, Callable
from starlette.background import BackgroundTasks as StarletteBackgroundTasks
from typing_extensions import Annotated, Doc, ParamSpec  # type: ignore [attr-defined]

        Add a function to be called in the background after the response is sent.

        Read more about it in the
        [FastAPI docs for Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/).
        