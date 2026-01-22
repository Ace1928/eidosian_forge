from typing import List
import datasets
from datasets.tasks import AudioClassification
from ..folder_based_builder import folder_based_builder
class AudioFolderConfig(folder_based_builder.FolderBasedBuilderConfig):
    """Builder Config for AudioFolder."""
    drop_labels: bool = None
    drop_metadata: bool = None