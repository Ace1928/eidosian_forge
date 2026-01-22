from typing import List
import datasets
from datasets.tasks import AudioClassification
from ..folder_based_builder import folder_based_builder
class AudioFolder(folder_based_builder.FolderBasedBuilder):
    BASE_FEATURE = datasets.Audio
    BASE_COLUMN_NAME = 'audio'
    BUILDER_CONFIG_CLASS = AudioFolderConfig
    EXTENSIONS: List[str]
    CLASSIFICATION_TASK = AudioClassification(audio_column='audio', label_column='label')