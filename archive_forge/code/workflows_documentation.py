import json
import os
from typing import Any, Dict, List, Optional, Union
import wandb
import wandb.data_types as data_types
from wandb.data_types import _SavedModel
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
Link the given model to a portfolio.

    A portfolio is a promoted collection which contains (in this case) model artifacts.
    Linking to a portfolio allows for useful model-centric workflows in the UI.

    Args:
        model: `_SavedModel` - an instance of _SavedModel, most likely from the output
            of `log_model` or `use_model`.
        target_path: `str` - the target portfolio. The following forms are valid for the
            string: {portfolio}, {project/portfolio},{entity}/{project}/{portfolio}.
        aliases: `str, List[str]` - optional alias(es) that will only be applied on this
            linked model inside the portfolio. The alias "latest" will always be applied
            to the latest version of a model.

    Returns:
        None

    Example:
        sm = use_model("my-simple-model:latest")
        link_model(sm, "my-portfolio")

    