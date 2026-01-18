import os
import json
import numpy as np
import ipywidgets as widgets
import pythreejs
import ipywebrtc
from IPython.display import display
def write_movie(self):
    with self.output:
        filename = self.filename_movie
        if not self.overwrite_video and os.path.exists(filename):
            name, ext = os.path.splitext(filename)
            i = 1
            filename = name + '_' + str(i) + ext
            while os.path.exists(filename):
                i += 1
                filename = name + '_' + str(i) + ext
        with open(filename, 'wb') as f:
            f.write(self.recorder.video.value)
        print('wrote', filename)