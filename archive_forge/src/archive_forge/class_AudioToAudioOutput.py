from typing import TYPE_CHECKING, List, TypedDict
class AudioToAudioOutput(TypedDict):
    """Dictionary containing the output of a [`~InferenceClient.audio_to_audio`] task.

    Args:
        label (`str`):
            The label of the audio file.
        content-type (`str`):
            The content type of audio file.
        blob (`bytes`):
            The audio file in byte format.
    """
    label: str
    content_type: str
    blob: bytes