import os
import pathlib
import tempfile
import uuid
import numpy as np
from ..utils import is_soundfile_availble, is_torch_available, is_vision_available, logging
class AgentAudio(AgentType):
    """
    Audio type returned by the agent.
    """

    def __init__(self, value, samplerate=16000):
        super().__init__(value)
        if not is_soundfile_availble():
            raise ImportError('soundfile must be installed in order to handle audio.')
        self._path = None
        self._tensor = None
        self.samplerate = samplerate
        if isinstance(value, (str, pathlib.Path)):
            self._path = value
        elif isinstance(value, torch.Tensor):
            self._tensor = value
        else:
            raise ValueError(f'Unsupported audio type: {type(value)}')

    def _ipython_display_(self, include=None, exclude=None):
        """
        Displays correctly this type in an ipython notebook (ipython, colab, jupyter, ...)
        """
        from IPython.display import Audio, display
        display(Audio(self.to_string(), rate=self.samplerate))

    def to_raw(self):
        """
        Returns the "raw" version of that object. It is a `torch.Tensor` object.
        """
        if self._tensor is not None:
            return self._tensor
        if self._path is not None:
            tensor, self.samplerate = sf.read(self._path)
            self._tensor = torch.tensor(tensor)
            return self._tensor

    def to_string(self):
        """
        Returns the stringified version of that object. In the case of an AgentAudio, it is a path to the serialized
        version of the audio.
        """
        if self._path is not None:
            return self._path
        if self._tensor is not None:
            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + '.wav')
            sf.write(self._path, self._tensor, samplerate=self.samplerate)
            return self._path