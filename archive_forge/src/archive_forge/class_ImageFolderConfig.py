from typing import List
import datasets
from datasets.tasks import ImageClassification
from ..folder_based_builder import folder_based_builder
class ImageFolderConfig(folder_based_builder.FolderBasedBuilderConfig):
    """BuilderConfig for ImageFolder."""
    drop_labels: bool = None
    drop_metadata: bool = None