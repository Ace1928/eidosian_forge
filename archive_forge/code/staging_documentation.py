import os
from wandb import env
from wandb.sdk.lib.filesystem import mkdir_exists_ok
from wandb.sdk.lib.paths import FilePathStr
Manages artifact file staging.

Artifact files are copied to the staging area as soon as they are added to an artifact
in order to avoid file changes corrupting the artifact. Once the upload is complete, the
file should be moved to the artifact cache.
