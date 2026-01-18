from .build import build
import os
def update_indexes(conversation):
    """
    Re-assigns indexes after smoothening (mostly for clarity purposes) Doesn't really
    matter since we never index by specifically using the "index" field of the json obj.

    :param conversation:
        The dialogue between USER and ASSISTANT with inconsistent indices
    """
    for i in range(len(conversation)):
        conversation[i]['index'] = i
    return conversation