from pathlib import Path
from typing import Any, List, Union
from langchain_community.document_loaders.unstructured import (


        Args:
            file_path: The path to the file to load.
            mode: The mode to use when loading the file. Can be one of "single",
                "multi", or "all". Default is "single".
            **unstructured_kwargs: Any kwargs to pass to the unstructured.
        