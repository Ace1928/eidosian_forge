from .build import build
import os

    Aggregates contiguous responses by the same speaker in the data so that data
    eventually contains alternations between USER and ASSISTANT.

    :param conversation:
        The dialogue between USER and ASSISTANT with possible multiple contiguous
        speeches by the same speaker
    :param opt:
        options dict, mainly useful for accessing value of exclude_invalid_data
    